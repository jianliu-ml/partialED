import pickle
import torch
import os
import numpy as np
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from transformers import BertForQuestionAnswering
from transformers import AdamW, get_linear_schedule_with_warmup

import config

from mrc_dataset import Dataset
from mrc_preprocessing import load_dataset
from utils import save_model, load_model

# https://nlp.jhu.edu/rams/

def build_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return model


def _get_target_index(token_to_orig_map, idx):
    result = []
    for i in token_to_orig_map:
        if token_to_orig_map[i] == idx:
            result.append(i)
    return result


def predict(model, device, data_set):
    model.eval()
    with torch.no_grad():

        predicted_s = []
        predicted_e = []
        appendixes = []

        for batch in data_set.get_tqdm(device, False):
            data_x, data_x_mask, data_segment_id, data_start, data_end, appendix = batch

            inputs = {  'input_ids': data_x,
                        'attention_mask':  data_x_mask,
                        'token_type_ids':  data_segment_id
                    }
            outputs = model(**inputs)
            
            predicted_s.extend(torch.softmax(outputs[0], -1).detach().cpu().numpy())
            predicted_e.extend(torch.softmax(outputs[1], -1).detach().cpu().numpy())
            appendixes.extend(appendix)

            # break
 
            # print(outputs[0][0])
        
        predicted_map = {}
        for app, s_prob, e_prob in zip(appendixes, predicted_s, predicted_e):

            # doc_offset, token_to_orig_map, sen_id, t, ev = app  # doc_offset, token_to_orig_map, sen_id, t, ev
            doc_offset, token_to_orig_map, t, doc_id, sen_id, words, pos, ners, events = app
            predicted_map.setdefault(sen_id, list())
            predicted_map[sen_id].append([s_prob, e_prob, t, events, token_to_orig_map, doc_offset])

        best = 0
        best_tuple = ()
        for th in range(0, 20):
            th = th/20
            n_correct, n_predict, n_golden = 0, 0, 0
            for sen_id in predicted_map:
                golden = [(x[0][0], x[0][2]) for x in predicted_map[sen_id][0][3]]
                predicted = []
                for elem in predicted_map[sen_id]:
                    s_prob, e_prob, t, ev, token_to_orig_map, doc_offset = elem
                    for idx in token_to_orig_map:
                        if s_prob[idx] > th:
                            predicted.append((token_to_orig_map[idx], t))

                golden = set(golden)
                predicted = set(predicted)
                inter = golden.intersection(predicted)

                n_predict += len(predicted)
                n_correct += len(inter)
                n_golden += len(golden)
            
            try:
                p = n_correct / n_predict
                r = n_correct / n_golden
                f1 = 2 * p * r / (p + r)
            except:
                p, r, f1 = 0, 0, 0
            if f1 > best:
                best = f1
                best_tuple = (th, p, r, f1)
        
        print(best_tuple)

    # return predicted_map



if __name__ == '__main__':
    device = 'cuda:0'
    batch_size = 100

    train_data, dev_data, test_data = load_dataset('data_add_mrc.pickle')
    print(len(train_data), len(dev_data), len(test_data))

    max_len = 150
    train_set = Dataset(batch_size, max_len, train_data)
    test_set = Dataset(batch_size, max_len, test_data)

    model = build_model(config.bert_dir)
    model = torch.nn.DataParallel(model)
    load_model(model, 'models/mrc_0.500000_1.pk')
    model.to(device)

    predict(model, device, test_set)
