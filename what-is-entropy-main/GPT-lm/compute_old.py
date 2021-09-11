import torch
import argparse
import subprocess
import sys
import numpy as np
import glob

from typing import (Iterable, List)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', type=str, default='', help='input data') # The torchtext way of loading data
    parser.add_argument('--output', '-o', type=str, default='', help='output file name')
    parser.add_argument('--input_pattern', '-ip', type=str, default='')
    parser.add_argument('--batch_size', '-bs', type=int, default=1)

    parser.add_argument('--model', type=str, 
        choices=['gpt', 'gpt2'], 
        default='gpt')

    return parser

def load_model(args):
    model_name = args.model
    if model_name == 'gpt':
        tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')
        model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
    elif model_name == 'gpt2':
        pass

    return tokenizer, model


def line_count(fname:str):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def get_lines(fname: str):
    with open(fname, 'r') as f:
        for line in f:
            yield line


def cross_entropy(probs, label: int):
    return - np.log(probs[label])

def entropy(probs):
    assert len(probs.shape) == 1
    # assert probs.dtype == np.float64
    
    probs_smoothed = probs.copy()
    size = probs_smoothed.shape[0]
    max_idx = np.argmax(probs_smoothed)
    left_idx = np.arange(max_idx)
    right_idx = np.arange(max_idx + 1, size)

    eps = np.finfo(np.float64).eps
    smooth = eps / (size - 1)
    probs_smoothed[max_idx] -= eps
    probs_smoothed[left_idx] += smooth

    entropy = - np.dot(probs_smoothed, np.log(probs_smoothed))

    return entropy, probs_smoothed

def process_batch(batch: List[str], tokenizer, model):
    """
    """
    tokenized_text = [tokenizer.tokenize(t) for t in batch]
    indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_text]
    actual_lens = list(map(len, indexed_tokens))
    max_len = max(actual_lens)

    unk_id = tokenizer.convert_tokens_to_ids('<unk>') # 0
    indexed_tokens_padded = [tokens + (max_len - len(tokens))*[unk_id] for tokens in indexed_tokens]
    tokens_tensor = torch.tensor(indexed_tokens_padded)

    with torch.no_grad():
        predictions = model(tokens_tensor)
    predictions = predictions[0]
    predictions_probs = torch.softmax(predictions, 2)

    cross_entropies = [
        [cross_entropy(predictions_probs[i, j, :].numpy(), int(tokens_tensor[i, j])) for j in range(actual_lens[i] - 1)] for i in range(predictions.size(0))
    ]
    entropies = [
        [entropy(predictions_probs[i, j, :].numpy()) for j in range(actual_lens[i] - 1)] for i in range(predictions.size(0))
    ]

    return cross_entropies, entropies


def process_file(fname, tokenizer, model, args, verbose=True):
    """
    Process a single input file, specified by args.input
    """
    cross_entropies, entropies = [], []
    for i, batch in enumerate(load_batch(fname, args.batch_size)):
        ce, e = process_batch(batch, tokenizer, model)
        cross_entropies += ce
        entropies += e
        if verbose:
            if i % 10 == 9:
                sys.stdout.write('\r{} batches processed'.format(i+1))
                sys.stdout.flush()
    
    return cross_entropies, entropies


def save_cross_entropies(fname: str, cross_entropies):
    with open(fname, 'w') as f:
        for i, ce in enumerate(cross_entropies):
            for j, it in enumerate(ce):
                f.write('{},{},{}\n'.format(i, j, it))

def save_entropies(fname: str, entropies):
    with open(fname, 'w') as f:
        for i, e in enumerate(entropies):
            for j, item in enumerate(e):
                ent, _ = item
                f.write('{},{},{}\n'.format(i, j, ent))

def load_batch(fname: str, batch_size: int) -> Iterable[List[str]]:
    """
    """
    batch: List[str] = []
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if len(batch) == batch_size:
                yield batch
                batch = [line]
            else:
                batch.append(line)
        if len(batch) > 0:
            yield batch


def main(args):
    tokenizer, model = load_model(args)

    # Process a single input file
    if args.input:
        ce, e = process_file(args.input, tokenizer, model, args, verbose=True)
        out1 = '{}_{}_cross_entropy.txt'.format(args.output, args.model)
        out2 = '{}_{}_entropy.txt'.format(args.output, args.model)
        save_cross_entropies(out1, ce)
        save_entropies(out2, e)

    # Process multiple input files that match the input_pattern
    elif args.input_pattern:
        fnames = glob.glob(args.input_pattern)
        for fn in fnames:
            print('processing {}'.format(fn))
            ce, e = process_file(fn, tokenizer, model, args, verbose=True)
            out1 = '{}_{}_cross_entropy.txt'.format(fn, args.model)
            out2 = '{}_{}_entropy.txt'.format(fn, args.model)
            save_cross_entropies(out1, ce)
            save_entropies(out2, e)
    

def test(args):
    count = 0
    for b in load_batch(args.input, args.batch_size):
        count += 1
    print(count)
    print(b)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)