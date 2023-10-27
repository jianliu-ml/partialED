import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import pickle

import config

from bio_preprocessing import load_dataset
from bio_dataset import Dataset
from bio_model import BertED
from conlleval import evaluate_conll_file
from utils import save_model, load_model

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def evaluate(model, device, test_set, filename='predict.conll'):
    model.eval()
    golds = []
    preds = []
    with torch.no_grad():
        for batch in test_set.get_tqdm(device, False):

            app = batch['app']
            golds.extend(app)

            logits = model.compute_logits(**batch)

            # classifications = torch.argmax(logits, -1)
            # classifications = list(classifications.cpu().numpy())

            logits = torch.softmax(logits, -1)
            result = []
            for batch_res in logits:
                temp = []
                arg_res = torch.argsort(batch_res, dim=1, descending=True)
                for arg_r, prob in zip(arg_res, batch_res):
                    t = arg_r[0].item()
                    if t == 0:
                        if prob[arg_r[1]] > 0.1:
                            t = arg_r[1].item()
                    temp.append(t)
                result.append(temp)
                    

                # temp = []
                # for elem in batch_res:

                #     t = torch.argmax(elem).item()
                #     if t == 0:
                #         max_i = -1
                #         max_pro = 0
                #         for i, e in enumerate(elem):
                #             if i == 0: continue
                #             if e > max_pro:
                #                 max_pro = e
                #                 max_i = i
                #         if max_pro > 0.3:
                #             t = max_i
                #     temp.append(t)

                # result.append(temp)
                        
            preds.extend(result)

    file_eval = open(filename, 'w')

    for gold, pred in zip(golds, preds):
        doc_id, idx, words, labels = gold[0], gold[1], gold[-2], gold[-1]
        pred = [config.idx2tag[x] for x in pred]

        print(idx, 'O', 'O', file=file_eval)
        for i, (w, l, p) in enumerate(zip(words, labels, pred)):
            l = 'B-'+l if l != 'O' else l
            p = 'B-'+p if p != 'O' else p
            print(doc_id + '|' + str(idx) + '|' + str(i) + '|' + w, l, p, file=file_eval)
        print(file=file_eval)

    file_eval.close()
    
    with open(filename) as fout:
        evaluate_conll_file(fout) 
    
    print()


if __name__ == '__main__':

    device = 'cuda'

    batch_size = 20

    train_data, dev_data, test_data = load_dataset('data_add_mrc.pickle')

    dev_data = list(filter(lambda x: len(x[2]) < 130, dev_data))

    # train_dataset = Dataset(batch_size, 180, train_data)
    dev_dataset = Dataset(batch_size, 130, dev_data)
    test_dataset = Dataset(batch_size, 130, test_data) 

    model = BertED(34, False, False)
    load_model(model, 'models/mrc_0.900000_4.pk')
    model.to(device)
    evaluate(model, device, test_dataset, 'test.conll')