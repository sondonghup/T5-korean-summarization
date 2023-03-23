import os
import json
import random
import re
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd

def load_dataset(data_dir, prefix, eos_token):

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    utterance_dict = {}
    text_list = []
    summary_list = []
    for file in os.listdir(data_dir):
        if file.endswith('csv'):
            datas = pd.read_csv(f'{data_dir}{file}')
            print(f'======= {file} =======')
            for id, text, summary in zip(tqdm(datas['id']), datas['text'], datas['summary']):
                text_list.append(text + prefix)
                summary_list.append(summary + eos_token)
            utterance_dict['text'] = text_list
            utterance_dict['summary'] = summary_list

    return utterance_dict

class DialogueDataset(Dataset):
    def __init__(self, datas, tokenizer) -> None:
        super(DialogueDataset, self).__init__()
        self.tokenizer = tokenizer
        self.input_ids = list()
        self.attention_mask = list()
        self.labels = list()

        print(f"model_max_length : {self.tokenizer.model_max_length}")

        for data, label in zip(tqdm(datas['text']), datas['summary']):
            encode_data = self.tokenizer(data, truncation = True, max_length = tokenizer.model_max_length)
            encode_label = self.tokenizer(label, truncation = True, max_length = tokenizer.model_max_length)

            self.input_ids.append(encode_data['input_ids'] + encode_label['input_ids'])
            self.attention_mask.append(encode_data['attention_mask'] + encode_label['attention_mask'])
            self.labels.append([-100] * len(encode_data['input_ids']) + encode_label['input_ids'])
            # print(f'\ninput_ids : {self.input_ids}')
            # print(f'\ninput_ids : {self.attention_mask}')
            # print(f'\ninput_ids : {self.labels}')
            # input()



    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

def collate_fn(batch, pad_token_id, bos_token_id):
    def seq_length_(p):
        return len(p[0])

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_seq_size = len(max_seq_sample)

    batch_size = len(batch)

    '''
    input_ids : pad_token_id로 패딩
    attention_masks : 0으로 패딩
    labels : -100으로 패딩
    '''

    input_ids = torch.zeros(batch_size, max_seq_size).fill_(pad_token_id).long()
    attention_masks = torch.zeros(batch_size, max_seq_size).fill_(0).long()
    labels = torch.zeros(batch_size, max_seq_size).fill_(-100).long()
    # input_ids = torch.full((batch_size, max_seq_size), pad_token_id).long()
    # attention_masks = torch.full((batch_size, max_seq_size), pad_token_id).long()
    # labels = torch.full((batch_size, max_seq_size), pad_token_id).long()

    for idx in range(batch_size):
        sample = batch[idx]
        sample_input_ids = sample[0]
        sample_attention_masks = sample[1]
        sample_labels = sample[2]

        '''
        y = tensor.new_tensor(x, requires_grad = True)
        -> 파라미터가 무엇이든 간에 이를 읽어서 leaf variable로 생성한다.
        y = x.clone.detach()
        -> computational graph에서 더이상 필요하지 않을 때 사용할 수 있다. computational graph에서 분리 할때
        -> y를 계산해도 x에 영향 없음 weight를 통해 특정 작업을 하고 싶을때 이용
        y = torch.empty_like(x).copy_(x)
        -> y에 gradient가 흐를수 있음
        y = torch.tensor(x)
        -> 명확하고 빠른 방법. 
        '''

        input_ids[idx].narrow(0, 0, len(sample_input_ids)).copy_(torch.LongTensor(sample_input_ids))
        attention_masks[idx].narrow(0, 0, len(sample_attention_masks)).copy_(torch.LongTensor(sample_attention_masks))
        labels[idx].narrow(0, 0, len(sample_labels)).copy_(torch.LongTensor(sample_labels))

    return input_ids, attention_masks, labels