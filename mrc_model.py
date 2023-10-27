import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import config

import numpy as np
from scipy.stats import entropy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertModel, BertPreTrainedModel

from mrc_preprocessing import load_dataset, dropout_dataset, retain_only_positive_examples
from mrc_dataset import Dataset


class BertForQuestionAnswering(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = True

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


def uncertainty_select(l): # less is better
    a = {}
    for e in l:
        a.setdefault(e, 0)
        a[e] += 1

    labels = [x for x in a]
    prob = [a[x]/len(l) for x in a]

    label = np.random.choice(labels, 1, p=prob)[0]
    return label


def uncertainty_interval(model, batch, N=10):
    model.train()
    result = []
    with torch.no_grad():
        data_x, data_x_mask, data_segment_id, data_start, data_end, appendix = batch
        inputs = {
            'input_ids': data_x,
            'attention_mask':  data_x_mask,
            'token_type_ids':  data_segment_id,   # Note Roberta does not use token_type_ids
            'start_positions': data_start,
            'end_positions':   data_end
            }
        for i in range(N):
            loss, start, end = model(**inputs)
            result.append([start, end])
    
    # print(result[0][0].size()) ## 20, 150
    # print(result[0][1].size()) ## 20, 150

    ## result -> (N, 2, batch_size, seq_len)

    for elem in result:
        elem[0] = torch.argmax(elem[0], 1).detach().cpu().numpy()
        elem[1] = torch.argmax(elem[1], 1).detach().cpu().numpy()
    ## result -> (N, 2, batch_size)

    result = np.asarray(result)
    result = np.transpose(result, (2, 1, 0))

    ## result -> (batch_size, 2, N)
    
    start = result[:,0,:]
    end = result[:,1,:]

    start = [uncertainty_select(x) for x in start]
    end = [uncertainty_select(x) for x in end]

    start = torch.LongTensor(start).to(data_x.device)
    end = torch.LongTensor(end).to(data_x.device)

    inputs = {
        'input_ids': data_x,
        'attention_mask':  data_x_mask,
        'token_type_ids':  data_segment_id,   # Note Roberta does not use token_type_ids
        'start_positions': start,
        'end_positions':   end
    }

    outputs = model(**inputs)
    return outputs



if __name__ == '__main__':

    l = [1, 1, 1, 2, 3, 4]
    prob = uncertainty_select(l)
    print(prob)


    device = 'cuda'
    org_train_data, dev_data, test_data = load_dataset('data_add_mrc.pickle')
    model = BertForQuestionAnswering.from_pretrained(config.bert_dir)
    model.to(device)

    train_set = Dataset(20, 200, org_train_data)
    for batch in train_set.get_tqdm(device, False):
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
        outputs = uncertainty_interval(model, batch)
        break