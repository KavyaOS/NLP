import tokenizers
import os
import sys
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_wordlevel_tokenizer(input_path, output_file, special_tokens = None):
    """
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-the-tokenizer
    We can set the training arguments like vocab_size or min_frequency (here left at their default values of 30,000 and 0) but the most important part is to give the special_tokens we plan to use later on (they are not used at all during training) so that they get inserted in the vocabulary.

    The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth.
    """
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) #TODO: not sure if this unk_token matters
    tokenizer.pre_tokenizer = Whitespace()

    if not special_tokens:
        special_tokens = []
    default_special_tokens = ['[UNK]', '[PAD]']
    for tok in default_special_tokens:
        if tok not in special_tokens:
            special_tokens.append(tok)
    trainer = WordLevelTrainer(special_tokens=special_tokens, min_frequency=20)

    # Check if the corpus data exist
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, split.csv) for split in ['train', 'test', 'valid']]
        for file in files:
            if not os.path.exists(file):
                print(f'{file} does not exist!')
                return
    elif os.path.isfile(input_path):
        if not os.path.exists(input_path):
            print(f'{input_path} does not exist!')
            return
        else:
            files = [input_path]

    tokenizer.train(files, trainer)
    tokenizer.save(output_file)


def test_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Single sentence
    output = tokenizer.encode("Hello, y'all!")
    print(output.tokens)

    # Batch of sentences
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
    outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(outputs[0].tokens)
    print(outputs[1].tokens)
    print(outputs[1].attention_mask())
    print(len(outputs))


def check_special_tokens(input_path):
    special_tokens = Counter()
    with open(input_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            for tok in tokens:
                if tok.startswith('<') and tok.endswith('>'):
                    special_tokens[tok] += 1
    print('==============')
    print(f'{len(special_tokens)} Special tokens detected:')
    for k, v in special_tokens.most_common():
        print(f'{k}: {v}')
    print('==============')
    for k, _ in special_tokens.most_common():
        sys.stdout.write(f'\'{k}\', ')
        sys.stdout.flush()


def main():
    # train_wordlevel_tokenizer('word-level-tokenizer-wiki2.json')
    # train_wordlevel_tokenizer(input_path='.data/WikiText2/wikitext-2/', output_file='word-level-tokenizer-wiki2_pad.json', special_tokens=['[UNK]', '[PAD]']) 
    # NOTE: the token in original corpus data is <unk>. If we use [UNK] as special_token, the resulting .json dictionary file still contains <unk> rather than [UNK]

    # check_special_tokens(input_path='data/ytb1_label.txt')
    # train_wordlevel_tokenizer(input_path='data/ytb1_label.txt', output_file='ytb1_label_pad.json', special_tokens=['<a>', '<b>', '<c>', '<d>', '<e>', '<f>', '<g>', '<h>'])

    # check_special_tokens(input_path='data/cleaned_data_final.txt')
    #train_wordlevel_tokenizer(input_path='data/position/cleaned_data_final.txt', output_file='data/position/cleaned_data_final_pad.json', special_tokens=['<70>', '<69>', '<43>', '<42>', '<40>', '<78>', '<39>', '<52>', '<51>', '<67>', '<79>', '<66>', '<48>', '<60>', '<61>', '<72>', '<129>'])
    
    check_special_tokens(input_path='data/combined_corpus/train.csv')
    train_wordlevel_tokenizer(input_path='data/combined_corpus/train.csv', output_file='results_after_freq_threshold/minimum_frequency_20/trained_tokenizers/combined_corpus.json', special_tokens=['<a>', '<b>', '<c>', '<d>', '<e>', '<f>', '<g>', '<h>'])

    #test_tokenizer('trained_tokenizers/bnc.json')


if __name__ == '__main__':
    main()