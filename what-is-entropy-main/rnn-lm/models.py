import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args):
        super(RNNModel, self).__init__()
        self.rnn_type = args.model
        self.ntoken = args.vocab_size
        self.ninp = args.emsize
        self.nhid = args.nhid
        self.nlayers = args.nlayers
        
        self.dropout = args.dropout
        self.drop = nn.Dropout(args.dropout)
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.ninp, self.nhid, self.nlayers, dropout=self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(self.ninp, self.nhid, self.nlayers, nonlinearity=nonlinearity, dropout=self.dropout)
        self.decoder = nn.Linear(self.nhid, self.ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if args.tied:
            if self.nhid != self.ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, x_lengths, h):
        """
        # Source: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # When training RNN (LSTM or GRU or vanilla-RNN), it is difficult to batch the variable length sequences. For example: if the length of sequences in a size 8 batch is [4,6,8,5,4,3,7,8], you will pad all the sequences and that will result in 8 sequences of length 8. You would end up doing 64 computations (8x8), but you needed to do only 45 computations. Moreover, if you wanted to do something fancy like using a bidirectional-RNN, it would be harder to do batch computations just by padding and you might end up doing more computations than required.
        # Instead, PyTorch allows us to pack the sequence, internally packed sequence is a tuple of two lists. One contains the elements of sequences. Elements are interleaved by time steps (see example below) and other contains the size of each sequence the batch size at each step. This is helpful in recovering the actual sequences as well as telling RNN what is the batch size at each time step. This has been pointed by @Aerin. This can be passed to RNN and it will internally optimize the computations.
        # Here's a code example:
        a = [torch.tensor([1,2,3]), torch.tensor([3,4])]
        b = torch.nn.utils.rnn.pad_sequence(a, batch_first=True)
        >>>>
        tensor([[ 1,  2,  3],
            [ 3,  4,  0]])
        torch.nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=[3,2])
        >>>>PackedSequence(data=tensor([ 1,  3,  2,  4,  3]), batch_sizes=tensor([ 2,  2,  1]))
        """
        emb = self.drop(self.encoder(x)) # dropout applied to embedding layer; shape: [bsize, maxseqlen, embsize], i.e., [B, L, E]

        emb_packed = nn.utils.rnn.pack_padded_sequence(emb, x_lengths, batch_first=True, enforce_sorted=False)
        out, h = self.rnn(emb_packed, h) # out shape: [B, L, H]
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = self.drop(out)
        decoded = self.decoder(out) # decoded shape: [B, L, V], V is vocabulary size
        probs = F.log_softmax(decoded, dim=1)

        return probs, h

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)