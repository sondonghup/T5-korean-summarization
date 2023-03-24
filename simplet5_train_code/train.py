from simplet5 import SimpleT5
import argparse
import os
import pandas as pd

def train(train_df, eval_df, output_dir):
    model = SimpleT5()

    model.from_pretrained("t5", "t5-base")

    model.train(train_df=train_df, # pandas dataframe with 2 columns: source_text & target_text
                eval_df=eval_df, # pandas dataframe with 2 columns: source_text & target_text
                source_max_token_len = 512, 
                target_max_token_len = 128,
                batch_size = 8,
                max_epochs = 5,
                use_gpu = True,
                outputdir = output_dir,
                early_stopping_patience_epochs = 0,
                precision = 32
                )
    
def load_data(dir):
    flies = os.listdir(dir)
    for file in flies:
        df = pd.read_csv(f'{dir}{file}')
        df.columns = ['id', 'source_text', 'target_text']
        df.drop(df.columns[0])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_model', type=str, default='lcw99/t5-large-korean-text-summary')
    parser.add_argument('-t', '--train_dir', type = str, default='/Users/sondonghyeob/Downloads/aiconnect/train/splited_train/')
    parser.add_argument('-v', '--valid_dir', type = str, default='/Users/sondonghyeob/Downloads/aiconnect/train/splited_valid/')
    parser.add_argument('-d', '--save_dir', type = str, default='/content/drive/MyDrive/gpt2_checkpoints/gpt2_checkpoints')
    parser.add_argument('-w', '--num_workers', type = int, default=4)
    parser.add_argument('-b', '--batch_size', type = int, default=10)
    parser.add_argument('-l', '--learning_rate', type = float, default=3e-5)
    parser.add_argument('-e', '--num_epochs', type = int, default=5)
    parser.add_argument('-g', '--gradient_clip_val', type = float, default=1.0 )
    parser.add_argument('-o', '--log_every', type=int, default=20)
    parser.add_argument('-a', '--accumulate_grad_batches', type = int, default=1)
    parser.add_argument('-s', '--save_every', type=int, default=10_000)
    parser.add_argument('-m', '--prefix', type=str, default='summarize: ')

    args = parser.parse_args()

    train_df = load_data(args.train_dir)
    valid_df = load_data(args.valid_dir)
    train(train_df, valid_df, args.save_dir)