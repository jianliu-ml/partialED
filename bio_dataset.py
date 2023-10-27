import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm

from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset, neg_ratio=None):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)
        self.neg_ratio = neg_ratio

    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        sub_word_lens = [len(tup[2]) for tup in batch]
        max_sub_word_len = self.seq_len

        original_sentence_len = [len(tup[5]) for tup in batch]
        max_original_sentence_len = self.seq_len

        data_x, data_span, data_y = list(), list(), list()
        appendix = list()

        for data in batch:
            data_x.append(data[2])
            data_span.append(data[3])
            
            if self.neg_ratio:
                temp = data[4][:]
                for i, _ in enumerate(temp):
                    if temp[i] == 0:
                        if random.random() < self.neg_ratio:
                            temp[i] = -1
                data_y.append(temp)
            else:
                data_y.append(data[4])

            appendix.append(data)

        f = torch.LongTensor

        data_x = list(map(lambda x: pad_sequence_to_length(x, max_sub_word_len), data_x))
        bert_mask = get_mask_from_sequence_lengths(f(sub_word_lens), max_sub_word_len)

        default_y = -1
        data_y = list(map(lambda x: pad_sequence_to_length(x, max_original_sentence_len, default_value=lambda: default_y), data_y))
        sequence_mask = get_mask_from_sequence_lengths(f(original_sentence_len), max_original_sentence_len)
        
        data_span_tensor = np.zeros((len(data_x), max_original_sentence_len, 2), dtype=int)
        for i in range(len(data_span)):
            temp = data_span[i] # [:len(appendix[i][5])]
            data_span_tensor[i, :len(temp), :] = temp

        return {
                'input_x': f(data_x).to(device),  
                'input_mask': bert_mask.to(device),
                'input_span': f(data_span_tensor).to(device),
                'seq_mask': sequence_mask.to(device),
                'label': f(data_y).to(device),
                'app': appendix
        }


if __name__ == '__main__':
    from bio_preprocessing import load_dataset
    train, dev, test = load_dataset()

    data_set = Dataset(20, 200, test, 0.3)
    for batch in data_set.reader('cpu', False):
        # print(batch['input_x'])
        print(batch['label'])