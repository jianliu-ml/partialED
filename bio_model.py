import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import BertModel

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from allennlp.modules.conditional_random_field import allowed_transitions, is_transition_allowed, ConditionalRandomField

from config import bert_dir, idx2tag

crf_trans = allowed_transitions('BIO', idx2tag)

class BertED(nn.Module):
    def __init__(self, y_num=None, top_rnns=False, use_crf=False):
        super().__init__()

        self.y_num = y_num
        self.bert = BertModel.from_pretrained(bert_dir, output_hidden_states=True)
       
        self.top_rnns=top_rnns
        if top_rnns:
            self.rnn = nn.LSTM(bidirectional=True, num_layers=1, input_size=self.bert.config.hidden_size, hidden_size=self.bert.config.hidden_size//2, batch_first=True)

        self.last_n_layer = 1
        self.span_extractor = EndpointSpanExtractor(input_dim=self.bert.config.hidden_size * self.last_n_layer, combination='x+y')
        # self.span_extractor = SelfAttentiveSpanExtractor(input_dim=self.bert.config.hidden_size * self.last_n_layer)

        self.fc = nn.Linear(self.bert.config.hidden_size * self.last_n_layer, y_num)
        
        self.use_crf=use_crf
        if use_crf:
            self.crf = ConditionalRandomField(len(idx2tag), crf_trans)
        
        self.dropout = nn.Dropout(0.5)
    
    def getName(self):
        return self.__class__.__name__

    def compute_logits(self, input_x, input_mask, input_span, *args, **param):

        outputs = self.bert(input_x, attention_mask=input_mask)
        
        bert_enc = outputs[0]
        hidden_states = outputs[2]
        
        # bert_enc = hidden_states[2]
        bert_enc = torch.cat(hidden_states[-self.last_n_layer:], dim=-1)

        logits = self.span_extractor(bert_enc, input_span)

        if self.top_rnns:
            logits, _ = self.rnn(logits)
        
        logits = self.fc(logits)

        return logits


    def forward(self, input_x, input_mask, input_span, seq_mask, label, *args, **param):
        
        logits = self.compute_logits(input_x, input_mask, input_span)

        ## Normal classification
        loss_fct = CrossEntropyLoss(ignore_index=-1) # -1 is pad
        loss = loss_fct(logits.view(-1, self.y_num), label.view(-1))

        if self.use_crf:
            loss = 0 - self.crf(logits, label, seq_mask)
        return loss
    

    def predict(self, data_x, bert_mask, data_span, sequence_mask):
        
        logits = self.compute_logits(data_x, bert_mask, data_span)
        
        classifications = torch.argmax(logits, -1)
        classifications = list(classifications.cpu().numpy())
        predicts = []
        for classification, mask in zip(classifications, sequence_mask):
            predicts.append(classification[:])

        if self.use_crf:
            predicts = self.crf.viterbi_tags(logits, sequence_mask)
            predicts = [x[0] for x in predicts]

        return predicts