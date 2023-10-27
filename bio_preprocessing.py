import pickle
import random
random.seed(123)
import config

tokenizer = config.tokenizer

def _bio_bert_examples(words, labels):
    subword_ids = list()
    spans = list()
    label_ids = list()

    for word, label in zip(words, labels):
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        s = len(subword_ids)
        subword_ids.extend(sub_tokens)
        e = len(subword_ids) - 1
        spans.append([s, e])

        label_ids.append(config.tag2idx[label])

    return subword_ids, spans, label_ids


def bio_examples(result):
    examples = []

    for idx, doc in enumerate(result):
        doc_id, words, pos, ners, events = doc
        labels = ['O'] * len(words)
        for event in events:
            s, e, event_type = event[0]
            labels[s] = event_type  ## single token ?

        subword_ids, spans, label_ids  = _bio_bert_examples(words, labels)
        examples.append([doc_id, idx, subword_ids, spans, label_ids, words, labels])

    return examples


def load_dataset(filename='data_bio.pickle'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dropout_dataset(data, remain_ratio=1):
    for elem in data:
        # print(elem[4])
        event_labels = elem[4]
        for i, e in enumerate(event_labels):
            if e != 0:
                if random.random() > remain_ratio:
                    elem[4][i] = 0
        # print(elem[4])
    return data


if __name__ == "__main__":

    data = load_dataset()
    train, dev, test = data
    train = dropout_dataset(train, 0.3)
    print(len(train))
    
