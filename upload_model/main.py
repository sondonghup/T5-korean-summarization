from transformers import (
    T5Tokenizer, T5ForConditionalGeneration
)
import argparse
import os
from huggingface_hub import HfApi

def model_upload(model_path, repo_name, commit_msg, commit_description):
    api = HfApi()
    # api.create_repo(repo_id = repo_name)    
    print('model loading...')

    repo_name = repo_name
    auth_token = os.environ['huggingface_token']
    files_to_push_to_hub = os.listdir(model_path)

    for filename in files_to_push_to_hub:
        api.upload_file(
            token=auth_token,
            path_or_fileobj=f"{model_path}{filename}",
            repo_id=repo_name,
            path_in_repo=filename,
            repo_type="model",
            create_pr=1,
            commit_message=commit_msg,
            commit_description=commit_description
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default='/Users/sondonghyeob/Downloads/simplet5-epoch-6-train-loss-0.1551-val-loss-0.1814/')
    parser.add_argument('--repo_name', type=str, default='sonacer/t5-base-ko-summarization')
    parser.add_argument('--commit_msg', type=str, default='add t5 model')
    parser.add_argument('--commit_description', type=str, default='t5 base ko model')

    args = parser.parse_args()

    model_upload(args.model_path, args.repo_name, args.commit_msg, args.commit_description)
