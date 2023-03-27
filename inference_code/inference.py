inference.py
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer
)
import re
from tqdm import tqdm
import pandas as pd
import os
from datetime import datetime, timezone, timedelta


def load_model_tokenizer(tokenizer_name, tokenizer_revision, model_name, model_revision):
    '''
    모델과 토크나이저와 gpu 로드

    tokenizer_name : 토크나이저 이름 (모델이름과 합치지 않은 이유는 허깅페이스 모델중에 토크나이저를 안 올리는 경우도 있어서 따로 분리)
    tokenizer_revision : 토크나이저 버전
    model_name : 모델 이름
    model_revision : 모델 버전
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TOKENIZER LOAD
    tokenizer = T5Tokenizer.from_pretrained(
    f'{tokenizer_name}', revision=f'{tokenizer_revision}'
    )

    # MODEL LOAD
    model = T5ForConditionalGeneration.from_pretrained(
    f'{model_name}', revision=f'{model_revision}',  
    ).to(device)

    # Evaluate Mode Setting
    model.eval()
    print('Inference is Ready')

    return tokenizer, model, device


class SummaryTestDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self._data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        prompt = "summarize : {text}"
        input_text = prompt.format(text=row['text'])
        input_encoding = self.tokenizer(input_text)

        result = {
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
        }
        
        return result

    def _left_pad(self, sequence, value, max_len):
        return [value] * (max_len - len(sequence)) + sequence

    def collate_fn(self, batch, device='cuda'):
        input_length = max(len(row['input_ids']) for row in batch)

        input_ids = [
            self._left_pad(row['input_ids'], self.tokenizer.pad_token_id, input_length)
            for row in batch
        ]
        attention_mask = [
            self._left_pad(row['attention_mask'], 0, input_length)
            for row in batch
        ]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long, device=device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long, device=device),
        }


def do_inference(model, tokenizer):
    '''
    test_lists : 전처리 된 test.csv
    type : 베이스라인에 있던 beam, greedy, topk등의 값이 유의미 하다 생각하여 분리 실험은 custom을 통해서 진행
    model : 모델
    tokenizer : 토크나이저
    device : cuda, cpu..
    '''
    preds = []
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                generated = model.generate(
                    input_ids = batch['input_ids'],
                    attention_mask = batch['attention_mask'],
                    pad_token_id = tokenizer.pad_token_id,
                    min_new_tokens = 20,
                    max_new_tokens = 100,
                    do_sample = False,
                    top_k=5,
                    temperature=1.05,
                    use_cache = True,
                )
            summary_tokens = generated
            summary = tokenizer.batch_decode(summary_tokens, skip_special_tokens=True)
            preds.extend(summary)
            print(*summary, sep='\n----------\n',end='\n========\n')

    return preds

if __name__ == "__main__":
    '''
    --type argument ex)
    greedy : greedy search
    beam : beam search without penalty
    beam_penalty : beam search with penalty
    topk : top k search without temperature
    topk_temperature : top k with temperature
    custom : 커스텀
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_name', type=str, default='lcw99/t5-large-korean-text-summary')
    parser.add_argument('--tokenizer_revision', type=str, default='main')
    parser.add_argument('--model_name', type=str, default='lcw99/t5-large-korean-text-summary')
    parser.add_argument('--model_revision', type=str, default='main')
    parser.add_argument('--input_dir', type=str, default='/content/drive/MyDrive/aiconnect/test/test.csv')
    parser.add_argument('--output_dir', type = str, default='/content/drive/MyDrive/aiconnect/submit/')

    args = parser.parse_args()

    target_tokenizer, target_model, device = load_model_tokenizer(args.tokenizer_name, args.tokenizer_revision, args.model_name, args.model_revision)
    test_set = SummaryTestDataset(args.input_dir, target_tokenizer)
    test_loader = DataLoader(test_set, batch_size=4, num_workers=0, shuffle=False, collate_fn=test_set.collate_fn)
    preds = do_inference(target_model, target_tokenizer)

    test_df = pd.read_csv(args.input_dir)
    test_df['summary'] = preds

    TIME_SERIAL = datetime.now(timezone(timedelta(hours=9))).strftime("%y%m%d-%H%M%S")
    SUBMISSION_PATH = f'/content/drive/MyDrive/aiconnect/submit/exp_{TIME_SERIAL}.csv'
    test_df[['id', 'summary']].to_csv(SUBMISSION_PATH, index=False)
    print(SUBMISSION_PATH)