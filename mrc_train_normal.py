import pickle
import torch
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

from transformers import AdamW, get_linear_schedule_with_warmup

import config

from mrc_dataset import Dataset
from mrc_preprocessing import load_dataset, dropout_dataset, retain_only_positive_examples
from mrc_evaluate import predict
from mrc_model import BertForQuestionAnswering
from utils import save_model, load_model


def build_model(model_path):
    model = BertForQuestionAnswering.from_pretrained(model_path)
    return model


from mrc_model import uncertainty_interval


if __name__ == '__main__':

    device = 'cuda'
    
    n_epoch = 5
    batch_size = 20 * len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

    org_train_data, dev_data, test_data = load_dataset('data_add_mrc.pickle')
    
    for portion in range(10, 11):
        portion = portion / 10
        print(portion)

        temp_train_data = copy.deepcopy(org_train_data)
        train_data = temp_train_data[: int(len(org_train_data) * 0.1)]

        print(portion, len(train_data))

        max_len = 150
        train_set = Dataset(batch_size, max_len, train_data)
        test_set = Dataset(batch_size, max_len, test_data)

        lr = 1e-5
        max_grad_norm = 1.0
        num_warmup_steps = 0
        num_training_steps = n_epoch * (len(train_data) / batch_size)
        warmup_proportion = float(num_warmup_steps) / float(num_training_steps)  # 0.1

        model = build_model(config.bert_dir)
        model = torch.nn.DataParallel(model)
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)  # PyTorch scheduler

        idx = 0
        for epo in range(n_epoch):
            step_in_batch = 0
            for batch in train_set.get_tqdm(device, False):
                idx += 1
                step_in_batch += 1
                model.train()
                data_x, data_x_mask, data_segment_id, data_start, data_end, appendix = batch

                inputs = {
                    'input_ids': data_x,
                    'attention_mask':  data_x_mask,
                    'token_type_ids':  data_segment_id,   # Note Roberta does not use token_type_ids
                    'start_positions': data_start,
                    'end_positions':   data_end
                    }

                outputs = model(**inputs)

                if step_in_batch % 20 == 0:
                    outputs = uncertainty_interval(model, batch)

                loss = outputs[0]
                loss = loss.mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            save_model(model, 'models/mrc_%f_%d.pk' % (portion, epo))
            predict(model, device, test_set)
