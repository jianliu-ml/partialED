import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import pickle
import copy

import config

from bio_preprocessing import load_dataset, dropout_dataset
from bio_dataset import Dataset
from bio_model import BertED
from bio_evaluate import evaluate
from conlleval import evaluate_conll_file
from utils import save_model

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


if __name__ == '__main__':

    device = 'cuda'

    lr = 1e-5
    batch_size = 20
    n_epochs = 10

    org_train_data, dev_data, test_data = load_dataset('data_add_bio.pickle')
    org_train_data = list(filter(lambda x: len(x[2]) < 180, org_train_data))

    print(len(org_train_data), len(dev_data), len(test_data))

    for ratio in range(1, 10):
        ratio = ratio / 10
        print('Ratio', ratio)

        temp_train_data = copy.deepcopy(org_train_data)
        train_data = temp_train_data[:int(len(org_train_data) * ratio)]

        ## train_dataset = Dataset(batch_size, 180, train_data, 0.3)

        train_dataset = Dataset(batch_size, 180, train_data)
        test_dataset = Dataset(batch_size, 130, test_data) 

        model = BertED(34, False, False)
        model.to(device)

        num_warmup_steps = 0
        num_training_steps = n_epochs * (len(train_data) / batch_size)
        warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = AdamW(parameters, lr=lr, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

        for idx in range(n_epochs):
            model.train()
            for batch in train_dataset.get_tqdm(device, True):
                loss = model(**batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            save_model(model, 'models/%d.ckp' % idx)
            evaluate(model, device, test_dataset)