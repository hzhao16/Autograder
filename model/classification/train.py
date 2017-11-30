import argparse
import time
import math
import numpy as np
import os 
import random
import pickle
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.insert(0, '../util')
import logger
import data
import classification

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=24,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
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
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--bidirectional', action='store_true', help='LSTM: bidirectional or not')
parser.add_argument("--log_path", type=str, default="../logs")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
logpath = os.path.join(args.log_path, args.model) + ".log"

logger = logger.Logger(logpath)

logger.Log(args)

###############################################################################
# Load data
###############################################################################

#data_set = data.load_data(args.data + '/topics_labeled_sample.xlsx')
#print('load data')
logger.Log('Load data')
data_set = data.load_data(args.data + '/training_set_rel3.tsv')
data_size = len(data_set)
#print('data_size', data_size)
training_set = data_set[:int(data_size*0.8)]
dev_set = data_set[int(data_size*0.8):int(data_size*0.9)]
test_set = data_set[int(data_size*0.9):]

word_to_ix, index_to_word, vocab_size = data.build_dictionary([training_set])
#print('vocab_size:', vocab_size)
data.sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set])

###############################################################################
# Training code
###############################################################################

def clip_gradient(model, clip):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        modulenorm = p.grad.data.norm()
        totalnorm += modulenorm ** 2
    totalnorm = math.sqrt(totalnorm)
    return min(1, args.clip / (totalnorm + 1e-6))


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# This is the iterator we'll use during training. 
# It's a generator that gives you one batch at a time.
def data_iter(source, batch_size):
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while True:
        start += batch_size
        if start > dataset_size - batch_size:
            # Start another epoch.
            start = 0
            random.shuffle(order)   
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        yield [source[index] for index in batch_indices]

# This is the iterator we use when we're evaluating our model. 
# It gives a list of batches that you can then iterate through.
def eval_iter(source, batch_size):
    batches = []
    dataset_size = len(source)
    start = -1 * batch_size
    order = list(range(dataset_size))
    random.shuffle(order)

    while start < dataset_size - batch_size:
        start += batch_size
        batch_indices = order[start:start + batch_size]
        batch = [source[index] for index in batch_indices]
        if len(batch) == batch_size:
            batches.append(batch)
        else:
            continue
        
    return batches

# The following function gives batches of vectors and labels, 
# these are the inputs to your model and loss function
def get_batch(batch):
    vectors = []
    labels = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["score"])
    return vectors, labels


def training_loop(batch_size, num_epochs, model, loss_, optim, training_iter, dev_iter, train_eval_iter):
    step = 0
    epoch = 0
    total_batches = int(len(training_set) / batch_size)
    hidden = model.init_hidden(batch_size)

    while epoch <= num_epochs:
        model.train()
        vectors, labels = get_batch(next(training_iter)) 
        #print(labels)
        vectors = Variable(torch.stack(vectors).squeeze())
        vectors = vectors.transpose(1, 0)
        labels = Variable(torch.stack(labels).squeeze())
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        output, hidden = model(vectors, hidden)
        #print('o',output.size())
        lossy = loss_(output, labels)
        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()
        
        if step % total_batches == 0:
            if epoch % 1 == 0:
                acc_train = evaluate(model, train_eval_iter)
                acc_dev = evaluate(model, dev_iter)
                logger.Log("Epoch %i; Step %i; Loss %f; Train acc: %f; Dev acc: %f" 
                      %(epoch, step, lossy.data[0], acc_train, acc_dev))
                #print("Epoch %i; Step %i; Loss %f; Train acc: %f; Dev acc: %f" 
                #      %(epoch, step, lossy.data[0], acc_train, acc_dev))
            epoch += 1
        step += 1

# This function outputs the accuracy on the dataset, we will use it during training.
def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0

    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable(torch.stack(vectors).squeeze())
        vectors = vectors.transpose(1, 0)
        labels = torch.stack(labels).squeeze()
        
        hidden = model.init_hidden(args.batch_size)
        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)

        total += labels.size(0)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).sum()
      
    return correct / float(total)


# This function gives us the confusion matrix for all labels and the overall accuracy.
def evaluate_confusion(model, data_iter):
    model.eval()
    correct_all = 0
    correct = {}
    for lab in easy_label_map:
        correct[lab] = [0,0,0,0,0] #eses, esar, ptpt, ptbr, total
    total = 0
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = Variable(torch.stack(vectors).squeeze())
        vectors = vectors.transpose(1, 0)
        labels = torch.stack(labels).squeeze()
        
        hidden = model.init_hidden(args.batch_size)
        hidden = repackage_hidden(hidden)
        output, hidden = model(vectors, hidden)
        
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct_all += (predicted == labels).sum()
        
        for lab in easy_label_map:
            inds = (labels[:] == easy_label_map[lab]).nonzero().squeeze()
            for i in range(len(easy_label_map)):
                tmp =  torch.ones(len(inds.size())).long()*i
                correct[lab][i] += (predicted[inds] == tmp).sum()
            correct[lab][-1] += inds.size(0)
        
        confusion = {}
        for val in correct:
            confusion[val] = {v:correct[val][i] for i, v in enumerate(easy_label_map)}
        
    return confusion, correct_all / float(total)

###############################################################################
# Load pre-trained embedding 
###############################################################################

matrix = np.zeros((2, int(args.emsize)))

oov=0
glove = {}
filtered_glove = {}
glove_path = '../../data/filtered_glove_{}.p'.format(int(args.emsize))
if(os.path.isfile(glove_path)):
    #print("Reusing glove dictionary to save time")
    logger.Log("Reusing glove dictionary to save time")
    pretrained_embedding = pickle.load(open(glove_path,'rb'))
else:
    if args.emsize in [50,100,200,300]:
        #print('loading glove embedding')
        with open('../../data/glove/glove.6B.{}d.txt'.format(int(args.emsize))) as f:
            lines = f.readlines()
            for l in lines:
               vec = l.split(' ')
               glove[vec[0].lower()] = np.array(vec[1:])
        #print('glove size={}'.format(len(glove)))
        #print("Finished making glove dictionary")

    for i in range(2, len(index_to_word)):
        word = index_to_word[i]
        if(word in glove):
            vec = glove[word]
            filtered_glove[word] = glove[word]
            matrix = np.vstack((matrix,vec))
        else:
            oov+=1
            random_init = np.random.uniform(low=-0.01,high=0.01, size=(1,args.emsize))
            matrix = np.vstack((matrix,random_init))

    pickle.dump(matrix, open("../../data/filtered_glove_{}.p".format(args.emsize), "wb"))
    #print(matrix.shape)
    pretrained_embedding = matrix
    #print("word_to_ix", len(word_to_ix))
    #print("oov={}".format(oov))
    #print("Saving glove vectors")
    logger.Log("Saving glove vectors")
###############################################################################
# Build the model
###############################################################################

# Build, initialize, and train model
rnn = classification.RNNModel(args.model, vocab_size, args.emsize, args.nhid, args.nlayers, dropout=args.dropout, bidirectional=args.bidirectional, pretrained_embedding=pretrained_embedding)
#rnn.init_weights()

# Loss and Optimizer
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr)

# Train the model
training_iter = data_iter(training_set, args.batch_size)
train_eval_iter = eval_iter(training_set, args.batch_size)
dev_iter = eval_iter(dev_set, args.batch_size)
#print('start training:')
logger.Log('start training:')
training_loop(args.batch_size, args.epochs, rnn, loss, optimizer, training_iter, dev_iter, train_eval_iter)

#dev_full_iter = eval_iter(dev_set, args.batch_size)
#print(evaluate_confusion(rnn, dev_full_iter))