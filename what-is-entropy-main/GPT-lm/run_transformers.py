import torch
import argparse
from transformers import (GPT2LMHeadModel, GPT2Tokenizer)
from tqdm import tqdm


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', help='input data') # The torchtext way of loading data
    parser.add_argument('--output', '-o', type=str, default='', help='output file name')
    parser.add_argument('--model', type=str, 
        choices=['gpt2'], 
        default='gpt2')

    return parser


def load_model(args):
    model_class = GPT2LMHeadModel
    tokenizer_class = GPT2Tokenizer
    pretrained_weights = 'gpt2'

    model = model_class.from_pretrained(pretrained_weights)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    model.eval()

    return model, tokenizer


def main(args):
    model, tokenizer = load_model(args)

    # Read lines from input
    results = []
    with open(args.input, 'r') as f:
        for line in tqdm(f.readlines(), ncols=100):
            tokens = tokenizer.tokenize(line.strip())
            indices = tokenizer.convert_tokens_to_ids(tokens)
            tensor = torch.tensor([indices])
            with torch.no_grad():
                loss, _, _ = model(tensor, labels=tensor)
            results.append(loss.item())
    
    # Write to output
    with open(args.output, 'w') as f:
        for res in results:
            f.write('{}\n'.format(res))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)