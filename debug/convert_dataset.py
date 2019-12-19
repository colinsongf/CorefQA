#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: convert_dataset.py
@time: 2019/12/02
@contact: wu.wei@pku.edu.cn
"""
import os
import json
from collections import Counter, defaultdict


def transform_tag(prev_tag: str) -> str:
    if prev_tag.startswith('M-') or prev_tag.startswith('E-'):
        return prev_tag.replace('M-', 'I-').replace('E-', 'I-')
    elif prev_tag.startswith('S-'):
        return prev_tag.replace('S-', 'B-')
    else:
        return prev_tag


def convert_conll2003(from_path, to_path):
    all_sents = []
    all_anns = []
    cur_sent = []
    cur_ann = []
    all_tags = Counter()
    with open(from_path) as fi:
        for i, line in enumerate(fi):
            if line.strip():
                token = line[0]
                ann = line[2:].strip()
                assert line[1] == ' '
                cur_sent.append(token)
                cur_ann.append(ann)
            else:
                all_sents.append(cur_sent)
                all_anns.append(cur_ann)
                cur_sent = []
                cur_ann = []
    print(len(all_sents))
    assert len(all_sents) == len(all_anns)
    with open(to_path, 'w') as fo:
        for sent, ann in zip(all_sents, all_anns):
            tags = [transform_tag(t) for t in ann]
            all_tags.update(tags)
            fo.write(json.dumps({'tokens': sent, 'tags': tags}, ensure_ascii=False) + '\n')
    print(all_tags)


def get_data_from_ann(ann_dir):
    for fname in os.listdir(ann_dir):
        if fname.endswith('.txt') and os.path.exists(os.path.join(ann_dir, fname.replace('.txt', '.ann'))):
            txt_file = open(os.path.join(ann_dir, fname)).read()
            indices = [i for i, c in enumerate(txt_file) if c == '\n']
            if 'negation' in open(os.path.join(ann_dir, fname.replace('.txt', '.ann'))).read():
                continue

            sentences = defaultdict(list)
            with open(os.path.join(ann_dir, fname.replace('.txt', '.ann'))) as fa:
                for annotation in fa:
                    if annotation.startswith('R'):
                        continue
                    tag_num, ann, text = annotation.split('\t')
                    tag_name, start, end = ann.split(' ')
                    start_index = max([i for i in indices if i <= int(start)])
                    end_index = min([i for i in indices if i >= int(end)])
                    sentences[txt_file[start_index: end_index]].append((tag_name, int(start) - start_index, int(end) - start_index, fname))
            print(sentences)


if __name__ == '__main__':
    # convert_conll2003('/data/nfsdata/nlp/xiaoya/per_org_gpe_title.char.bmes.clean',
    #                   '/data/nfsdata/nlp/datasets/ner/policy_brain_crawl.jsonl')
    # convert_conll2003('/data/nfsdata/nlp/datasets/sequence_labeling/CN_NER/CompanyNameAnn/test.char.bmes',
    #                   '/data/nfsdata/nlp/datasets/ner/news_company_test.jsonl')
    get_data_from_ann('/data/nfs/jingrong/share/star-market-prophet/annotated_data/star_market_1_1000')
