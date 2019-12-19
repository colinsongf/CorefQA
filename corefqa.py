"""使用问答做指代消解"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import random
import threading

import numpy as np
import tensorflow as tf

import conll
import coref_ops
import metrics
import optimization
import util
from bert import modeling
from bert import tokenization
from pytorch_to_tf import load_from_pytorch_checkpoint


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.max_segment_len = config['max_segment_len']
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        self.subtoken_maps = {}
        self.gold = {}
        self.eval_data = None  # Load eval data lazily.
        self.dropout = None
        self.bert_config = modeling.BertConfig.from_json_file(config["bert_config_file"])
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config['vocab_file'], do_lower_case=False)

        input_props = []
        input_props.append((tf.int32, [None, None]))  # input_ids. (batch_size, seq_len)
        input_props.append((tf.int32, [None, None]))  # input_mask (batch_size, seq_len)
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None, None]))  # Speaker IDs.  (batch_size, seq_len)
        input_props.append((tf.int32, []))  # Genre.  能确保整个batch都是同主题，能因为一篇文章的多段放在一个batch里
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts. 一个instance只有一个start?是整篇文章的所有mention的start
        input_props.append((tf.int32, [None]))  # Gold ends. 整篇文章的所有mention的end
        input_props.append((tf.int32, [None]))  # Cluster ids. 整篇文章的所有mention的id
        input_props.append((tf.int32, [None]))  # Sentence Map 整篇文章的每个token属于哪个句子

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)  # 10是batch_size?
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()  # self.queue_input_tensors 不一样？

        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        # bert stuff
        tvars = tf.trainable_variables()
        # If you're using TF weights only, tf_checkpoint and init_checkpoint can be the same
        # Get the assignment map from the tensorflow checkpoint.
        # Depending on the extension, use TF/Pytorch to load weights.
        assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, config[
            'tf_checkpoint'])
        init_from_checkpoint = tf.train.init_from_checkpoint if config['init_checkpoint'].endswith('ckpt') else load_from_pytorch_checkpoint
        init_from_checkpoint(config['init_checkpoint'], assignment_map)
        print("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            # init_string)
            print("  name = %s, shape = %s%s" % (var.name, var.shape, init_string))

        num_train_steps = int(self.config['num_docs'] * self.config['num_epochs'])  # 文章数 * 训练轮数
        num_warmup_steps = int(num_train_steps * 0.1)  # 前1/10做warm_up
        self.global_step = tf.train.get_or_create_global_step()  # 根据不同的model得到不同的optimizer
        self.train_op = optimization.create_custom_optimizer(tvars, self.loss, self.config['bert_learning_rate'],
                                                             self.config['task_learning_rate'],
                                                             num_train_steps, num_warmup_steps, False, self.global_step,
                                                             freeze=-1, task_opt=self.config['task_optimizer'],
                                                             eps=config['adam_eps'])
        self.coref_evaluator = metrics.CorefEvaluator()

    def start_enqueue_thread(self, session):
        """多线程读json文件准备训练数据"""
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:  # 每个例子是一篇文章，同一篇文章的所有段落一起做session run
                random.shuffle(train_examples)
                if self.config['single_example']:
                    for example in train_examples:
                        tensorized_example = self.tensorize_example(example, is_training=True)
                        feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)
                else:  # 所有文章切出来的段落整体shuffle再一个一个session run
                    examples = []
                    for example in train_examples:
                        tensorized = self.tensorize_example(example, is_training=True)
                        if type(tensorized) is not list:
                            tensorized = [tensorized]
                        examples += tensorized
                    random.shuffle(examples)
                    print('num examples', len(examples))
                    for example in examples:
                        feed_dict = dict(zip(self.queue_input_tensors, example))
                        session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session):
        """在evaluate和predict阶段加载整个模型"""
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables()]
        saver = tf.train.Saver(vars_to_restore)
        checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
            if s not in speaker_dict and len(speaker_dict) < self.config['max_num_speakers']:
                speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training):
        """
        把一篇文章的所有原始特征信息，分片转成tensor
        :param example:
        :param is_training:
        :return:
        """
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}  # 每个mention的token span对应的mention id
        cluster_ids = np.zeros(len(gold_mentions))  # 每个mention_id对应的cluster_id
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]  # 多少个滑动窗口
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        # assert num_words == len(speakers), (num_words, len(speakers))
        speaker_dict = self.get_speaker_dict(util.flatten(speakers))
        sentence_map = example['sentence_map']  # 每个token_id对应的sentence_id，标出句子的边界，防止出现跨句的candidate_span
        max_sentence_length = self.max_segment_len
        text_len = np.array([len(s) for s in sentences])  # 滑动窗口每个滑块的长度

        input_ids, input_mask, speaker_ids = [], [], []
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
            while len(sent_input_ids) < max_sentence_length:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            speaker_ids.append(sent_speaker_ids)
            input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)  # 多个滑动窗口，每个滑动窗口有一个list的tokens
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        doc_key = example["doc_key"]  # mention span是以sub-token为基准的，但评测时是以token为基准的
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)  # sub-token对回原来是第几个单词
        self.gold[doc_key] = example["clusters"]  #
        genre = self.genres.get(doc_key[:2], 0)

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                           cluster_ids, sentence_map)
        """
        input_ids: [3, 128] 切成块的三份input_ids
        input_mask: [3, 128] 每个token对应的0，1 mask
        text_len: [3, ] 切成块的三份各自的长度
        speaker_ids: [3, 128] 每个token对应的speaker_id
        genre: int 整篇文章的genre
        is_training: 是否在training阶段
        gold_starts: [23, ] 每个gold_mention的start_index
        gold_ends: [23, ] 每个gold_mention的end_index
        cluster_ids: [23, ] 每个gold_mention对应的cluster_id
        sentence_map: [257, ] 每个token在原来的第几个句子里
        """
        if is_training and len(sentences) > self.config["max_training_sentences"]:
            if self.config['single_example']:
                return self.truncate_example(*example_tensors)
            else:
                offsets = range(self.config['max_training_sentences'], len(sentences),
                                self.config['max_training_sentences'])
                tensor_list = [self.truncate_example(*(example_tensors + (offset,))) for offset in offsets]
                return tensor_list
        else:
            return example_tensors

    def truncate_example(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends,
                         cluster_ids, sentence_map, sentence_offset=None):
        """因为显存装不下，训练时对于一篇文章的所有128的doc_span，随机选其中连续的n个做训练，mentions也做相应的截取"""
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0,
                                         num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        # candidate_starts, candidate_ends: [num_candidates, ] 候选span的start_index, end_index
        # labeled_starts, labeled_ends: [num_mentions, ] 真实span的start_index, end_index
        # labels: [num_mentions, ] 每个gold_mention对应的cluster_id
        # same_span: [num_labeled, num_candidates] 哪些预测跟真实span完全一致，如果predict_i == label_j则c[i, j]=1否则为0
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0))
        same_span = tf.logical_and(same_start, same_end)
        # candidate_labels: [num_candidates] 预测对的candidate标上正确的cluster_id，预测错的标0
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels  # 每个候选答案得到真实标注的cluster_id

    def get_dropout(self, dropout_rate, is_training):  # is_training为True时keep=1-drop, 为False时keep=1
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_pruning(self, top_span_emb, top_span_mention_scores, c):
        """在取出的前k个候选span，针对每个span取出前c个antecedent，其mention score得分的组成是
        1. 每个span的mention score
        2. emb_i * W * emb_j的得分
        3. 每个span只取前面的span作为antecedent
        4. span与antecedent的距离映射为向量算个分
        """
        k = util.shape(top_span_emb, 0)  # num_candidates
        top_span_range = tf.range(k)  # [num_candidates, ]
        # antecedent_offsets: [num_candidates, num_candidates] 每两个span之间的距离，隔了几个span
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0)
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]
        if self.config['use_prior']:
            antecedent_distance_buckets = self.bucket_distance(antecedent_offsets)  # [k, k]
            distance_scores = util.projection(tf.nn.dropout(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), self.dropout), 1,
                initializer=tf.truncated_normal_initializer(stddev=0.02))  # [10, 1]
            antecedent_distance_scores = tf.gather(tf.squeeze(distance_scores, 1), antecedent_distance_buckets)  # [k,k]
            fast_antecedent_scores += antecedent_distance_scores
        # 取fast_antecedent_score top_k高的antecedent，每个antecedent对应的span_index
        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c] 每个pair对应的mask
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c] 每个pair对应的score
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c] 每个pair对应的offset
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_predictions_and_loss(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts,
                                 gold_ends, cluster_ids, sentence_map):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            use_one_hot_embeddings=False,
            scope='bert')
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        mention_doc = model.get_sequence_output()  # (batch_size, seq_len, hidden)
        mention_doc = self.flatten_emb_by_sentence(mention_doc, input_mask)  # (b, s, e) -> (b*s, e) 取出有效token的emb
        num_words = util.shape(mention_doc, 0)  # b*s

        # candidate_span: 每个位置都可能是起点，对每个起点有max_span_width种不同的终点，总共有(num_words, max_span_width)种可能
        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width])
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0)

        # [num_words, max_span_width]，根据index将对应位置的sentence_id取出来
        candidate_start_sentence_indices = tf.gather(sentence_map, candidate_starts)
        candidate_end_sentence_indices = tf.gather(sentence_map, tf.minimum(candidate_ends, num_words - 1))
        # [num_words, max_span_width]，合法的span需要满足start/end不能越界；start/end必须在同一个句子里
        candidate_mask = tf.logical_and(candidate_ends < num_words,
                                        tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices))
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        # [num_candidates] 把候选span mask掉再铺平
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]), flattened_candidate_mask)
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates] 每个候选span的cluster_id
        # [num_candidates, emb] 候选答案的向量表示  [num_candidates,] 候选答案的得分
        candidate_span_emb = self.get_span_embmax_top_antecedents(mention_doc, candidate_starts, candidate_ends)
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)

        # beam size 所有span的数量小于num_words * top_span_ratio
        k = tf.minimum(3900, tf.to_int32(tf.floor(tf.to_float(num_words) * self.config["top_span_ratio"])))
        c = tf.minimum(self.config["max_top_antecedents"], k)  # 初筛挑出0.4*500=200个候选，细筛再挑出50个候选
        # pull from beam，光使用mention_score卡前0.4*num_words个span
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   num_words,
                                                   True)  # [1, k]
        top_span_indices = tf.reshape(top_span_indices, [-1])  # k个按mention_score初筛出来的candidate的index

        # 取出top_k的span的信息，过coarse的span pair筛选，每个span取前c个antecedent
        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]

        # def body(idx, tensors):
        #     fake_input = tf.stack([top_span_starts, top_span_ends])
        #     fake_model = modeling.BertModel(
        #         config=self.bert_config,
        #         is_training=is_training,
        #         input_ids=fake_input,
        #         use_one_hot_embeddings=False,
        #         scope='bert')
        #     fake_output = fake_model.get_sequence_output()
        #     return idx + 1, tf.Print(tensors, [tf.shape(fake_output)], 'fake_output')
        #
        # # do the loop:
        # initial_outs = model.get_sequence_output()
        # _, final_outs = tf.while_loop(lambda z, t: z < 100, body, loop_vars=(0, initial_outs))
        # top_span_emb = tf.Print(top_span_emb, [tf.shape(tf.stack(final_outs))], "final_outs")
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_pruning(
            top_span_emb, top_span_mention_scores, c)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]],
                                              initializer=tf.truncated_normal_initializer(stddev=0.02)), genre)  # [emb]
        if self.config['use_metadata']:
            speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask)  # 拍平后加mask
            top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # 每个span取start位置的speaker_id
        else:
            top_span_speaker_ids = None

        dummy_scores = tf.zeros([k, 1])  # [k, 1]

        num_segs, seg_len = util.shape(input_ids, 0), util.shape(input_ids, 1)
        word_segments = tf.tile(tf.expand_dims(tf.range(0, num_segs), 1), [1, seg_len])
        flat_word_segments = tf.boolean_mask(tf.reshape(word_segments, [-1]), tf.reshape(input_mask, [-1]))
        # mention_segments：[num_candidates, ] 找出每个candidate_span在第几个segment里
        mention_segments = tf.expand_dims(tf.gather(flat_word_segments, top_span_starts), 1)  # [k, 1]
        # antecedent_segments: [k, c] 找出每个candidate_span的每个antecedents对应在第几个segment里
        antecedent_segments = tf.gather(flat_word_segments, tf.gather(top_span_starts, top_antecedents))  # [k, c]
        segment_distance = None
        if self.config['use_segment_distance']:  # [k, c] 每个mention和其antecedent之间隔了几个segment
            segment_distance = tf.clip_by_value(mention_segments - antecedent_segments, 0,
                                                self.config['max_training_sentences'] - 1)
        if self.config['fine_grained']:  # 所谓融入high-order information
            for i in range(self.config["coref_depth"]):
                with tf.variable_scope("coref_layer", reuse=(i > 0)):
                    top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                    top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(
                        top_span_emb,
                        top_antecedents,
                        top_antecedent_emb,
                        top_antecedent_offsets,
                        top_span_speaker_ids,
                        genre_emb,
                        segment_distance)  # [k, c] 算出最后的得分s(i, j) =sm(i) + sm(j) + sc(i, j) + sa(i, j)
                    # top_antecedent_weights： [k, c + 1] 每个mention对所有antecedent分配权重
                    # top_antecedent_emb：[k, c + 1, emb] 每个mention每个antecedent的embedding
                    # attended_span_emb：[k, emb] 每个mention所有antecedent的表示做加权和
                    top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1))
                    top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1)
                    attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1)
                    with tf.variable_scope("f"):
                        f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                       util.shape(top_span_emb, -1)))  # [k, emb]
                        top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]
        else:
            top_antecedent_scores = top_fast_antecedent_scores

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        # top_antecedent_cluster_ids [k, c] 每个mention每个antecedent的cluster_id
        # same_cluster_indicator [k, c] 每个mention跟每个预测的antecedent是否同一个cluster
        # pairwise_labels [k, c] 用pairwise的方法得到的label，非mention、非antecedent都是0，mention跟antecedent共指是1
        # top_antecedent_labels [k, c+1] 最终的标签，如果某个mention没有antecedent就是dummy_label为1
        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]
        # top_antecedent_labels = tf.Print(top_antecedent_labels, [tf.shape(top_antecedent_labels)], "ant labels")
        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedents, top_antecedent_scores], loss

    def get_span_emb(self, context_outputs, span_starts, span_ends):
        """一个span的表示由以下部分组成：
        span_start_embedding, span_end_embedding, span_width_embedding, head_attention_representation
        """
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = span_width - 1  # [k]
            span_width_emb = tf.gather(
                tf.get_variable("span_width_embeddings", [self.config["max_span_width"], self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
            head_attn_reps = tf.matmul(mention_word_scores, context_outputs)  # [K, T]
            span_emb_list.append(head_attn_reps)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb, span_starts, span_ends):
        with tf.variable_scope("mention_scores"):  # [k, 1] 每个候选span的得分
            span_scores = util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)
        if self.config['use_prior']:
            span_width_emb = tf.get_variable("span_width_prior_embeddings",
                                             [self.config["max_span_width"], self.config["feature_size"]],
                                             initializer=tf.truncated_normal_initializer(stddev=0.02))  # [W, emb]
            span_width_index = span_ends - span_starts  # [NC]
            with tf.variable_scope("width_scores"):
                width_scores = util.ffnn(span_width_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                         self.dropout)  # [W, 1]
            width_scores = tf.gather(width_scores, span_width_index)
            span_scores += width_scores
        return tf.squeeze(span_scores, 1)

    def get_width_scores(self, doc, starts, ends):
        distance = ends - starts
        span_start_emb = tf.gather(doc, starts)
        hidden = util.shape(doc, 1)
        with tf.variable_scope('span_width'):
            span_width_emb = tf.gather(
                tf.get_variable("start_width_embeddings", [self.config["max_span_width"], hidden],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)), distance)  # [W, emb]
        scores = tf.reduce_sum(span_start_emb * span_width_emb, axis=1)
        return scores

    def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
        num_words = util.shape(encoded_doc, 0)  # T
        num_c = util.shape(span_starts, 0)  # NC
        doc_range = tf.tile(tf.expand_dims(tf.range(0, num_words), 0), [num_c, 1])  # [num_candidates, num_words]
        mention_mask = tf.logical_and(doc_range >= tf.expand_dims(span_starts, 1),
                                      doc_range <= tf.expand_dims(span_ends, 1))  # [num_candidates, num_words]
        with tf.variable_scope("mention_word_attn"):
            word_attn = tf.squeeze(
                util.projection(encoded_doc, 1, initializer=tf.truncated_normal_initializer(stddev=0.02)), 1)
        mention_word_attn = tf.nn.softmax(tf.log(tf.to_float(mention_mask)) + tf.expand_dims(word_attn, 0))
        return mention_word_attn  # [num_candidates, num_words]

    def softmax_loss(self, antecedent_scores, antecedent_labels):
        # antecedent_scores: [k, c+1] 模型算出来的mention pair的得分
        # antecedent_labels: [k, c+1] 真实的mention pair的标签
        # 目标是最大化共指的mention pair的得分，也即最大化共指的antecedent的边缘概率
        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # 标签为0score为负无穷；标签为1score为预测score
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
        loss = log_norm - marginalized_gold_scores  # [k]
        return tf.reduce_sum(loss)

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)

    def get_mutual_information(self, pairwise_score):
        """
        1/2 * p(a2|a1) + 1/2 * p(a1/a2)
        :param pairwise_score:
        :return:
        """
        reversed_score = tf.transpose(pairwise_score, 1, 2)
        return 0.5 * pairwise_score + 0.5 * reversed_score

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb, segment_distance=None):
        """用更多的特征对mention-pair做细筛，用到的特征有：
        1. 两个mention的speaker是否相同
        2. genre对应的embedding
        3. 两个mention之间隔了几个mention
        4. 两个mention之间隔了几个segment
        将上述特征和头尾两个span的embedding拼在一起过全连接，得到mention-pair的得分
        """
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            # 只看当前candidate跟antecedent是不是same speaker
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]],
                                                         initializer=tf.truncated_normal_initializer(stddev=0.02)),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]],
                                initializer=tf.truncated_normal_initializer(stddev=0.02)),
                antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)
        if segment_distance is not None:
            with tf.variable_scope('segment_distance', reuse=tf.AUTO_REUSE):
                segment_distance_emb = tf.gather(tf.get_variable("segment_distance_embeddings",
                                                                 [self.config['max_training_sentences'],
                                                                  self.config["feature_size"]],
                                                                 initializer=tf.truncated_normal_initializer(
                                                                     stddev=0.02)), segment_distance)  # [k, emb]
            feature_emb_list.append(segment_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):  # emb_i * W * emb_j 算的分
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        """
        获得每个mention最可能的antecedent
        :param antecedents: (k, c) 每个mention对应antecedents的index
        :param antecedent_scores: (k, c+1) 每个mention和其对应antecedents之间的得分
        :return:
        """
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):  # 每个mention只找分数最大的
            if index < 0:  # 如果没有一个mention的antecedent分数大于零，认为他没有共指
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        """
        根据antecedent关系获得cluster信息
        :param top_span_starts: [k, ] 每个span在token级别的start_index
        :param top_span_ends: [k, ] 每个span在token级别的end_index
        :param predicted_antecedents: [k, ] 每个span在span级别的antecedent的index
        :return:
        """
        mention_to_predicted = {}  # mention的(start_idx, end_idx)到cluster_id的映射
        predicted_clusters = []  # cluster_id到mention的(start_idx, end_idx)的映射
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:  # 这个mention找不到antecedent
                continue
            assert i > predicted_index, (i, predicted_index)  # 从后往前找
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:  # predicted_antecedent如果之前见过了，就拿它所在的cluster_id
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)  # 如果没见过，新建一个cluster_id
                predicted_clusters.append([predicted_antecedent])  # cluster -> mention
                mention_to_predicted[predicted_antecedent] = predicted_cluster  # mention -> cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))  # 该mention的表示
            predicted_clusters[predicted_cluster].append(mention)  # cluster -> mention
            mention_to_predicted[mention] = predicted_cluster  # mention -> cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]  # mention 到 共指的所有mention的映射
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        self.coref_evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):  # lazy加载，真正需要evaluate的时候加载，加载一次常驻内存
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            # num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data) 所有token数
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False, eval_mode=False):
        self.load_eval_data()
        coref_predictions = {}

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, gold_starts, gold_ends, _, _ = tensorized_example
            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, \
                top_antecedents, top_antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)
            """
            candidate_starts: (num_words, max_span_width) 所有候选span的start
            candidate_ends: (num_words, max_span_width) 所有候选span的end
            candidate_mention_scores: (num_candidates,) 候选答案的得分
            top_span_starts: (k, ) 筛选过mention之后的候选的start_index
            top_span_ends: (k, ) 筛选过mention之后的候选的end_index
            top_antecedents: (k, c) 粗筛过antecedent之后的每个候选antecedent的index
            top_antecedent_scores: (k, c) 粗筛过antecedent之后的每个候选antecedent的score
            """
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"])
            if (example_num + 1) % 100 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        summary_dict = {}
        if eval_mode:  # 在测试集评测的时候，需要用官方的脚本再评测一遍
            conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions,
                                                 self.subtoken_maps, official_stdout)
            average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
            summary_dict["Average F1 (conll)"] = average_f1
            print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = self.coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(self.eval_data)))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util.make_summary(summary_dict), f
