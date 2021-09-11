import tokenizers
import os
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def train_wordlevel_tokenizer(input_path, output_file, prefix = 'wiki', special_tokens = ['<unk>', '[PAD]']):
    """
    https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-the-tokenizer
    We can set the training arguments like vocab_size or min_frequency (here left at their default values of 30,000 and 0) but the most important part is to give the special_tokens we plan to use later on (they are not used at all during training) so that they get inserted in the vocabulary.

    The order in which you write the special tokens list matters: here "[UNK]" will get the ID 0, "[CLS]" will get the ID 1 and so forth.
    """
    tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(special_tokens=["<unk>", '[PAD]'])

    # Check if the corpus data exist
    files = [os.path.join(input_path, f'{prefix}.train.tokens') for split in ['train', 'test', 'valid']]
    for file in files:
        if not os.path.exists(file):
            print(f'{file} does not exist!')
            return

    tokenizer.train(files, trainer)
    tokenizer.save(output_file)


def test_tokenizer(tokenizer_path: str):
    tokenizer = Tokenizer.from_file('word-level-tokenizer-wiki2.json')

    # Single sentence
    output = tokenizer.encode("Hello, y'all!")
    print(output.tokens)

    # Batch of sentences
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
    outputs = tokenizer.encode_batch(["Hello, y'all!", "How are you 😁 ?"])
    print(outputs[0].tokens)
    print(outputs[1].tokens)


def main():
    # train_wordlevel_tokenizer('word-level-tokenizer-wiki2.json')
    train_wordlevel_tokenizer(input_path='.data/WikiText2/wikitext-2/', output_file='word-level-tokenizer-wiki2_pad.json', special_tokens=['[UNK]', '[PAD]']) 
    # NOTE: the token in original corpus data is <unk>. If we use [UNK] as special_token, the resulting .json dictionary file still contains <unk> rather than [UNK]


if __name__ == '__main__':
    main()