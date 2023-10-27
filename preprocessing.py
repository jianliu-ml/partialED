import pickle

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

    # tokens.append(sep_token)
    # segment_ids.append(sequence_a_segment_id)

    for i in range(len(all_doc_tokens)):
        token_to_orig_map[len(tokens)] = tok_to_orig_index[i]
        tokens.append(all_doc_tokens[i])
        segment_ids.append(sequence_b_segment_id)

    # SEP token
    # tokens.append(sep_token)
    # segment_ids.append(sequence_b_segment_id)

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



def to_examples(data, max_seq_len=100):
    result = []
    for elem in data:
        doc_id, words, _, ners, events = elem
        input_ids, input_mask, segment_ids, _, token_to_orig_map, _, _ = build_bert_example('', words, -1, -1, max_seq_len)
        labels = [0] * len(config.event_set)
        for e in events:
            x, y, t = e[0]
            idx = config.tag2idx[t]
            for i in token_to_orig_map:
                if i < max_seq_len and token_to_orig_map[i] == x:
                    labels[idx] = i
                    # multiple?
        result.append([input_ids, input_mask, segment_ids, labels, token_to_orig_map])
    return result


def load_dataset():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    with open('data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    
    train, dev, test = data['train'], data['val'], data['test']
    print(len(train), len(dev), len(test))

    max_seq_len = 100
    train = to_examples(train, max_seq_len)
    dev = to_examples(dev, max_seq_len)
    test = to_examples(test, max_seq_len)

    data = [train, dev, test]

    for i, elem in enumerate(data):
        print(elem)
        i += 1
        if i == 0:
            break

    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    

    # from dataset import Dataset
    # from model import QueryEE

    # my_model = QueryEE(768)

    # train_set = Dataset(20, 100, test)

    # for batch in train_set.reader('cpu', False):
    #     data_x, data_x_mask, data_segment_id, data_y, appendix = batch
    #     my_model(data_x, data_x_mask, data_segment_id, data_y)
    #     break
