import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

#num_labels = 13

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, nn_type, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5, bidirectional = False, pretrained_embedding=None):
        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = getattr(nn, rnn_type)(embedding_dim, hidden_size, num_layers, bias=False, dropout=dropout, bidirectional=bidirectional)
        self.decoder = nn.Linear(hidden_size, 1)
        self.decoder_bi = nn.Linear(hidden_size*2, 1)

        self.init_weights()

        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def init_weights(self, pretrained_embedding=None):
        initrange = 0.1
        if(pretrained_embedding is not None):
            pretrained_embedding = pretrained_embedding.astype(np.float32)
            pretrained_embedding = torch.from_numpy(pretrained_embedding)
            if self.args.cuda:
                pretrained_embedding = pretrained_embedding.cuda()
            self.encoder.weight.data = pretrained_embedding
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden):
        emb = self.encoder(inputs)
        output, hidden = self.rnn(emb, hidden)
        output = torch.mean(output, 0)
        output = torch.squeeze(output)
        if self.bidirectional:
            decoded = self.decoder_bi(output)
        else:
            decoded = self.decoder(output)
        
        #decoded = self.sigmoid(decoded)
        return decoded, hidden
    
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM' and self.bidirectional == True:
            return (Variable(weight.new(self.num_layers * 2, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers * 2, bsz, self.hidden_size).zero_()))
        elif self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_()))            
        else:
            return Variable(weight.new(self.num_layers, bsz, self.hidden_size).zero_())