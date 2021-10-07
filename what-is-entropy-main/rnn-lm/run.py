# from main import repackage_hidden
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from utils import Corpus
import models

import argparse
import os
import time
import math

# print(torchtext.__version__)


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='.data/WikiText2/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--tokenizer', type=str, default='word-level-tokenizer-wiki2.json',
                    help='pretrained tokenizer')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')

parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')

parser.add_argument('--lr', type=float, default=0.05,
                    help='initial learning rate')
parser.add_argument('--lr_scheduler', type=str, choices=['StepLR', 'ExponentialLR'], default='')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')

parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')

# For experiments
parser.add_argument('--task', type=str, default='main', choices=['main', 'compute'])
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--output_ppl', type=str, default='', help='path to save the perplexity (ppl) scores of test data')

args, unknown = parser.parse_known_args()

###
# Initialization
###

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")


###
# Load data
###
corpus = Corpus(args.data, prefix='wiki.', ext='.tokens')
train_data, val_data, test_data = corpus.get_data()

# Load tokenizer
tokenizer = Tokenizer.from_file(args.tokenizer)
# NOTE: Make sure the tokenizer loaded from file has the correct special tokens

tokenizer.enable_padding(pad_id=1, pad_token="[PAD]") #NOTE: use a different pad_id other than -1
# get vocab_size from tokenizer, including the padding token
vars(args)['vocab_size'] = tokenizer.get_vocab_size() + 1 # +1 for [PAD]
# print(tokenizer.id_to_token(1)) # it is the word `the`, so using pad_id=1 is not okay


def collate_batch(batch):
    batch = [item.strip() for item in batch] # NOTE:  `if len(item.strip())>0` This condition is not safe, it changes the size of the actual batch. Consider moving it to the previous step, i.e., the raw iter in Corpus class.
    encoded_results = tokenizer.encode_batch(batch)

    ids_list, attn_mask_list, lengths_list = [], [], []
    for res in encoded_results:
        ids_list.append(res.ids)
        attn_mask_list.append(res.attention_mask)
        lengths_list.append(res.attention_mask.count(1)) # mask is of 1s and 0s, the count of 1s is the actual length

    ids = torch.tensor(ids_list, dtype=torch.int64)
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.int64)
    lengths = torch.tensor(lengths_list, dtype=torch.int64)

    return ids, attention_mask, lengths


train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch) # shuffle=False will cause error
val_loader = DataLoader(val_data, batch_size=40, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_data, batch_size=40, shuffle=False, collate_fn=collate_batch)

def process_batch(batch, pad_id = 1, flat_target = True):
    """
    params:
        batch - itered item from train_loader
        flat_target: True for training, False for computing perplexity per sentence
    returns:
        (data, data_lengths, target) 
    """
    # target is the same as the input_ids
    ids, mask, lengths = batch
    pads = torch.full((ids.shape[0], 1), pad_id)
    ids_shifted = torch.cat((ids, pads), dim=1) # Use torch.cat to add a column of all pad_id to the right of ids
    targets = ids_shifted[:, 1:]
    if flat_target:
        targets = ids.view(-1)# targets () is a flat tensor

    return (ids, lengths, targets) 


def repackage_hidden(h, device):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach().to(device)
    else:
        return tuple(repackage_hidden(v, device) for v in h)


###
# Define the model
###
model = models.RNNModel(args).to(device)
criterion = nn.NLLLoss(ignore_index=-1) # ignore the [pad] token
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr)

# Setup learning rate decay
if args.lr_scheduler:
    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif args.lr_scheduler == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

###
# Training 
###
def train(train_loader, epoch: int = 1):
    model.train()
    total_loss = 0.
    start_time = time.time()

    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for step, batch in enumerate(train_loader):
        data, data_lengths, targets = process_batch(batch)
        data = data.to(device)
        data_lengths = data_lengths # lengths must be on CPU
        targets = targets.to(device)

        model.zero_grad()
        if args.model != 'Transformer':
            if data.shape[0] != args.batch_size: # For the last batch's actual size is not necessarily args.batch_size
                hidden = model.init_hidden(data.shape[0])
            hidden = repackage_hidden(hidden, device)
            try:
                output, hidden = model(data, data_lengths, hidden)
            except RuntimeError:
                # Print debug info
                print('data.shape:', data.shape)
                print('data: ', data)
                print('data_lengths.shape:', data_lengths.shape)
                if isinstance(hidden, tuple):
                    print(hidden[0].shape)
                    print(hidden[1].shape)
                raise
        else:
            output = model(data, data_lengths)

        output_flat = output.view(-1, args.vocab_size) # In order to match the size of `targets`, which is also a flat tensor
        try:
            loss = criterion(output_flat, targets)
            loss.backward()
        except RuntimeError:
            # Print debug info
            print(output.shape)
            print('target shape: ', targets.shape)
            print('data shape: ', data.shape)
            raise

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        optimizer.step()

        total_loss += loss.item()
        if step % args.log_interval == 0 and step > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, step, len(train_loader), args.lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

            # Debug info
            # print('output_flat shape: ', output_flat.shape)
            # print('output shape: ', output.shape)
            # v = torch.sum(torch.exp(output.detach()), dim=2)
            # print('v shape', v.shape)
            # print('v', v)
        step += 1


###
# Evaluate 
###
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch in val_loader:
        data, data_lengths, targets = process_batch(batch)
        data = data.to(device)
        targets = targets.to(device)
        if args.model != 'Transformer':
            if data.shape[0] != args.batch_size: # For the last iter in an epoch, the actual size of data is not necessarily args.batch_size
                hidden = model.init_hidden(data.shape[0])
            hidden = repackage_hidden(hidden, device)
            output, hidden = model(data, data_lengths, hidden)
        else:
            output = model(data, data_lengths)
        loss = criterion(output.view(-1, args.vocab_size), targets)
        total_loss += loss.item()

    total_loss /= len(val_loader)

    return total_loss


###
# A standalone function for computing and outputing perplexity scores
###
@torch.no_grad()
def ppl_by_sentence(model, loader):
    model.eval()
    fwriter = None
    if args.output_ppl:
        fwriter = open(args.output_ppl, 'w')
    
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    for batch in loader:
        data, data_lengths, targets = process_batch(batch, flat_target=False) # should not flat the output
        data = data.to(device)
        targets = targets.to(device)
        if args.model != 'Transformer':
            if data.shape[0] != args.batch_size: # For the last iter in an epoch, the actual size of data is not necessarily args.batch_size
                hidden = model.init_hidden(data.shape[0])
            hidden = repackage_hidden(hidden, device)
            output, hidden = model(data, data_lengths, hidden)
        else:
            output = model(data, data_lengths)
    
        if args.output_ppl:
            for i in range(output.shape[0]):
                o = output[i]
                t = targets[i]
                ppl = criterion(o, t) # 
                fwriter.write(str(ppl.item()))
                fwriter.write('\n')
    if fwriter:
        fwriter.close()


def main():
    global model
    best_val_loss = None
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(train_loader, epoch)
            val_loss = evaluate(model, val_loader)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(model, test_loader)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)


def compute():
    with open(args.load, 'rb') as f:
        model = torch.load(f)
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()
    ppl_by_sentence(model, test_loader)
    # evaluate(model, test_loader)


if __name__ == '__main__':
    if args.task == 'main':
        main()
    elif args.task == 'compute':
        compute()
