import torch
import numpy as np
from typing import List, Tuple, Dict
import subprocess
import sys


def demo():
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')

    # text = "Who was Jim Henson ? Jim Henson was a puppeteer"
    text = 'Pierre Vinken, 61 years old, will join the board as a'
    tokenized_text = tokenizer.tokenize(text)
    print(text)
    print(tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    print(tokens_tensor.size(1))

    ### Predict the next token using `openAIGPTLMHeadModel`
    #######################################################
    lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
    lm_model.eval()

    with torch.no_grad():
	    predictions = lm_model(tokens_tensor)
    print('predictions[0].size(): ', predictions[0].size())
    
    # One-shot prediction
    print('Oneshot prediction:')
    for i in range(len(tokenized_text)):
        # print(predictions[0][0,i,:].size())
        predicted_index = torch.argmax(predictions[0][0, i, :]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print('predicted token at position {}: {}'.format(i, predicted_token))
    
    # Predict incrementally
    print('Incremental prediction:')
    for i in range(len(tokenized_text)):
        # i = 5
        sub_tensor = tokens_tensor[:,:(i+1)]
        # print(sub_tensor)
        with torch.no_grad():
            pred = lm_model(sub_tensor)
            predicted_index = torch.argmax(pred[0][0, -1, :]).item()
            predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
            print('predicted token at position {}: {}'.format(i, predicted_token))
    #NOTE: good news is, one-shot prediction generates exactly the same results as incremental prediction.
    
    # Original experiment
    predicted_index = torch.argmax(predictions[0][0, -1, :]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # assert predicted_token == '.</w>'

    print('The most likely word after the sentence \"{}\" is: \"{}\"'.format(text, predicted_token))


def demo_mult_sents():
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')
    lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
    lm_model.eval()

    text1 = "Who was Jim Henson ? Jim Henson was a super nice"
    text2 = 'Pierre Vinken, 61 years old, will join the board as a nonexecutive director Nov. 29'
    text3 = 'Rudolph Agnew, 55 years old and former chairman of Consolidated Gold Fields PLC, was named a nonexecutive director of this British industrial conglomerate'
    text = [text1, text2, text3]

    tokenized_text = [tokenizer.tokenize(t) for t in text]
    print(tokenized_text)
    indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_text]
    print(indexed_tokens)

    UNK_ID = tokenizer.convert_tokens_to_ids('<unk>') # 0
    actual_lens = list(map(len, indexed_tokens))
    max_len = max(actual_lens)
    indexed_tokens_padded = [tokens + (max_len - len(tokens))*[UNK_ID] for tokens in indexed_tokens]
    print('max_len:', max_len)
    print('indexed_tokens_padded', indexed_tokens_padded)

    tokens_tensor = torch.tensor(indexed_tokens_padded)
    print(tokens_tensor)
    print('tokens_tensor size: ', tokens_tensor.size())

    with torch.no_grad():
        predictions = lm_model(tokens_tensor)
    predictions = predictions[0]
    predictions_probs = torch.softmax(predictions, 2)
    print('predictions size: {}'.format(predictions.size()))
    
    last_predicted_indices = [torch.argmax(predictions[i, actual_lens[i]-1, :]).item() for i in range(len(text))]
    print(last_predicted_indices)
    last_predicted_tokens = tokenizer.convert_ids_to_tokens(last_predicted_indices)
    print(last_predicted_tokens)

    # Test cross_entropy
    cross_entropies = [
            [cross_entropy(predictions_probs[i, j, :].numpy(), tokens_tensor[i, j]) for j in range(actual_lens[i] - 1)] for i in range(predictions.size(0))
        ]
    print(cross_entropies)
    print('cross_entropies size:', list(map(len, cross_entropies)))


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


def process_wsj():
    # Load tokenizer and model
    tokenizer = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTTokenizer', 'openai-gpt')
    lm_model = torch.hub.load('huggingface/pytorch-pretrained-BERT', 'openAIGPTLMHeadModel', 'openai-gpt')
    lm_model.eval()
    UNK_ID = tokenizer.convert_tokens_to_ids('<unk>') # 0

    def process_batch(batch: List[str]):
        tokenized_text = [tokenizer.tokenize(t) for t in batch]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(t) for t in tokenized_text]
        actual_lens = list(map(len, indexed_tokens))
        max_len = max(actual_lens)
        indexed_tokens_padded = [tokens + (max_len - len(tokens))*[UNK_ID] for tokens in indexed_tokens]
        tokens_tensor = torch.tensor(indexed_tokens_padded)

        with torch.no_grad():
            predictions = lm_model(tokens_tensor)
        predictions = predictions[0]
        predictions_probs = torch.softmax(predictions, 2)

        cross_entropies = [
            [cross_entropy(predictions_probs[i, j, :].numpy(), int(tokens_tensor[i, j])) for j in range(actual_lens[i] - 1)] for i in range(predictions.size(0))
        ]
        entropies = [
            [entropy(predictions_probs[i, j, :].numpy()) for j in range(actual_lens[i] - 1)] for i in range(predictions.size(0))
        ]
        return cross_entropies, entropies

    text_file = '../ngram-lm/wsj_raw_utf8.txt'
    batch_size = 1
    cross_entropies, entropies = [], []
    with open(text_file, 'r') as f:
        text_batch = []
        for i, text in enumerate(f):
            if len(text_batch) == batch_size:
                ce, e = process_batch(text_batch)
                cross_entropies += ce
                entropies += e
                text_batch = [text]
            else:
                text_batch.append(text)
            if i % 10 == 0:
                sys.stdout.write('\r{} lines processed'.format(i+1))
                sys.stdout.flush()
        if len(text_batch) > 0:
            ce = process_batch(text_batch)
            cross_entropies += ce
            entropies += e
    
    output_file = text_file + '.GPT_cross_entropy'
    with open(output_file, 'w') as f:
        for i, ce in enumerate(cross_entropies):
            for j, it in enumerate(ce):
                f.write('{},{},{}\n'.format(i, j, it))
    
    output_file2 = text_file + '.GPT_entropy'
    with open(output_file2, 'w') as f:
        for i, e in enumerate(entropies):
            for j, item in enumerate(e):
                ent, _ = item
                f.write('{},{},{}\n'.format(i, j, ent))


def convert(input_file, output_file):
    with open(input_file, 'r') as fr, open(output_file, 'w') as fw:
        for i, line in enumerate(fr):
            items = line.strip().split(',')
            for j, it in enumerate(items):
                fw.write('{},{},{}\n'.format(i, j, it))


if __name__ == "__main__":
    # demo()
    # demo_mult_sents()
    process_wsj()