import json


def print_clusters(data_file):
    all_tokens = []
    all_mentions = []
    mention_len = []
    f = open(data_file)
    for i, line in enumerate(f):
        data = json.loads(line)
        tokens = [i for j in data['sentences'] for i in j]
        mentions = [i for j in data['clusters'] for i in j]
        all_tokens.append(len(tokens))
        all_mentions.append(len(mentions))
        mention_len.extend([m[1] - m[0] for m in mentions])
        print(len(mentions) / len(tokens))
    return all_tokens, all_mentions, mention_len


if __name__ == '__main__':
    tokens, mentions, m_len = print_clusters('/data/nfsdata2/home/wuwei/study/coref/data/test.english.128.jsonlines')
    print(sorted(m_len))
