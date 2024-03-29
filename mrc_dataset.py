import copy
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

class Dataset(object):
    def __init__(self, batch_size, seq_len, dataset):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.construct_index(dataset)

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
        if shuffle:
            self.shuffle()
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        # doc_id, role_name, input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position
        # input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position, sen, ev
        
        sentence_lens = [len(data[0]) for data in batch]
        max_sentence_len = self.seq_len

        data_x, data_x_mask, data_segment_id, data_start, data_end = list(), list(), list(), list(), list()
        appendix = list()

        for data in batch:
            input_ids, input_mask, segment_ids, doc_offset, token_to_orig_map, start_position, end_position, t, doc_id, idx, words, pos, ners, events = data
            data_x.append(input_ids)
            data_x_mask.append(input_mask)
            data_segment_id.append(segment_ids)
            data_start.append(start_position)
            data_end.append(end_position)

            appendix.append([doc_offset, token_to_orig_map, t, doc_id, idx, words, pos, ners, events])

        f = torch.LongTensor

        data_x = f(data_x)
        data_x_mask = f(data_x_mask).to(float)
        data_segment_id = f(data_segment_id)
        data_start = f(data_start)
        data_end = f(data_end)


        return [data_x.to(device),  
                data_x_mask.to(device),
                data_segment_id.to(device),
                data_start.to(device),
                data_end.to(device),
                appendix]


if __name__ == '__main__':
    from mrc_preprocessing import load_dataset
    train, dev, test = load_dataset()
    print(test[0])

    data_set = Dataset(20, 200, test)
    for batch in data_set.reader('cpu', False):
        print(batch[0])