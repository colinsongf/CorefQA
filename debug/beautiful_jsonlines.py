#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: beautiful_jsonlines.py
@time: 2019/10/05
@contact: wu.wei@pku.edu.cn

将json lines转成可读性强的形式
"""
import os
import json
from collections import Counter, defaultdict


DIR = '/data/nfsdata2/home/wuwei/study/e2e-coref'


def beautify_json(res):
    all_mentions = set()
    beautiful_sents = []
    beautiful_coref = []
    index_map = {}
    overall_index = 0
    for i, sent in enumerate(res['sentences']):
        beautiful_sents.append(' '.join(sent) + ' SPEAKER ' + res['speakers'][i][0])
        beautiful_coref.append(['_'] * len(sent))
        for j, word in enumerate(sent):
            index_map[overall_index] = (i, j)
            overall_index += 1
    for c_id, cluster in enumerate(res['clusters']):
        cluster_indexes = set()
        for mention in cluster:
            if tuple(mention) not in all_mentions:
                all_mentions.add(tuple(mention))
            else:
                print(res['doc_key'], mention)
            for overall_id in range(mention[0], mention[1] + 1):
                sent_id, word_id = index_map[overall_id]
                beautiful_coref[sent_id][word_id] = str(c_id)
                if overall_id not in cluster_indexes:
                    cluster_indexes.add(overall_id)
                else:
                    print(res['doc_key'], mention)
    return beautiful_sents, beautiful_coref


def convert_all():
    for fname in os.listdir(DIR):
        if fname.endswith('.jsonlines'):
            print(fname)
            with open(os.path.join(DIR, fname)) as fi, open(os.path.join(DIR, fname+'_readable'), 'w') as fo:
                for line in fi:
                    data = json.loads(line.strip())
                    sents, corefs = beautify_json(data)
                    fo.write('\n\n' + data['doc_key'] + '\n')
                    for sent, coref in zip(sents, corefs):
                        fo.write(sent + '\n')
                        fo.write('    '.join(coref) + '\n\n')


def get_dist():
    # file_types = defaultdict(list)
    speakers = []
    number_of_clusters = []
    length_of_clusters = []
    with open(os.path.join(DIR, 'dev.english.jsonlines')) as fi:
        for line in fi:
            data = json.loads(line.strip())
            file_type = data["doc_key"].split('/')[0]
            num_speakers = len(set([i for j in data['speakers'] for i in j]))
            num_clusters = len(data['clusters'])
            cluster_lens = [len(c) for c in data['clusters']]
            speakers.append(num_speakers)
            number_of_clusters.append(num_clusters)
            length_of_clusters.extend(cluster_lens)
    return Counter(speakers)


if __name__ == '__main__':
    print(get_dist())
