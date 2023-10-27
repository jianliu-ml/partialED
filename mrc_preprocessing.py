import json
import pickle
import random
import copy

import config

tokenizer = config.tokenizer

def build_bert_example(query, context, start_pos, end_pos, max_seq_length,
                    cls_token='[CLS]', sep_token='[SEP]', pad_token=0, 
                    sequence_a_segment_id=0, sequence_b_segment_id=1,
                    cls_token_segment_id=0, pad_token_segment_id=1):
    
    is_impossible = (start_pos == -1)
    query_tokens = tokenizer.tokenize(query)

    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    tok_to_orig_index = []
    orig_to_tok_index = []

    all_doc_tokens = []
    for (i, token) in enumerate(context):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)

        if len(all_doc_tokens) + len(sub_tokens) > max_tokens_for_doc:
            break

        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = -1
    tok_end_position = -1
    try:
        tok_start_position = orig_to_tok_index[start_pos]    # the answer is not in the current window
        tok_end_position = orig_to_tok_index[end_pos]
    except:
        tok_start_position = -1
        tok_end_position = -1

    tokens = []
    token_to_orig_map = {}
    segment_ids = []

    tokens.append(cls_token)
    segment_ids.append(cls_token_segment_id)
    cls_index = 0

    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(sequence_a_segment_id)

    tokens.append(sep_token)
    segment_ids.append(sequence_a_segment_id)

    for i in range(len(all_doc_tokens)):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        tokens.append(all_doc_tokens[i])
        segment_ids.append(sequence_b_segment_id)

    # SEP token
    tokens.append(sep_token)
    segment_ids.append(sequence_b_segment_id)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(pad_token)
        input_mask.append(0)
        segment_ids.append(pad_token_segment_id)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    
    doc_offset = len(query_tokens) + 2

    if is_impossible:
        start_position = cls_index
        end_position = cls_index
    else:
        start_position = tok_start_position + doc_offset
        end_position = tok_end_position + doc_offset

    return input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position

    
def mrc_examples(data, max_len=100, train=True):
    examples = []
    for idx, doc in enumerate(data):
        doc_id, words, pos, ners, events = doc
            
        for t in config.event_set:
            if train:
                flag = False
                for e in events:
                    start, end, event_type = e[0]
                    if t == event_type:
                        query = t.replace('_', ' ').lower()
                        exp = build_bert_example(query, words, start, end, max_len)
                        examples.append(list(exp) + [t, doc_id, idx, words, pos, ners, events])
                        flag = True
                if not flag:
                    query = t.replace('_', ' ').lower()
                    exp = build_bert_example(query, words, -1, -1, max_len)
                    examples.append(list(exp) + [t, doc_id, idx, words, pos, ners, events])
                    
            else:
                query = t.replace('_', ' ').lower()
                exp = build_bert_example(query, words, -1, -1, max_len)
                examples.append(list(exp) + [t, doc_id, idx, words, pos, ners, events])

    return examples


def load_dataset(filename='data_mrc.pickle'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def dropout_dataset(data, remain_ratio=1):
    res = []

    drop_set = {}
    idx_role_set = {}
    for elem in data:
        flag = True
        s, t = elem[5], elem[6]
        role, idx = elem[7], elem[9]

        if s != 0:
            if random.random() > remain_ratio: ## how to exclude the data?
                one_example = elem[:]
                one_example[5], one_example[6] = 0, 0
                drop_set[(role, idx)] = one_example  # record one elem in case all the examples of a role are removed ...
                continue

        idx_role_set.setdefault(idx, set())
        idx_role_set[idx].add(role)
        res.append(elem)
    
    for elem in idx_role_set:
        for r in config.event_set:
            if not r in idx_role_set[elem]:
                res.append(drop_set[(r, elem)])
    return res


def retain_only_positive_examples(data, ratio=0.01):
    res = []
    for elem in data:
        s, t = elem[5], elem[6]
        if s != 0:
            res.append(elem)
        else:
            if random.random() < ratio:
                res.append(elem)        
    return res



if __name__ == '__main__':
    
    data = load_dataset()
    train, dev, test = data
    print(len(train))
    train = dropout_dataset(train, 0.1)
    print(len(train))
    train = retain_only_positive_examples(train)
    print(len(train))
    
    