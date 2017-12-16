import argparse
import time
import math
import random
import numpy as np
import os 
import sys
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

sys.path.insert(0, '../util')
import logger
import data
import regression
import pickle

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--name', type=str, default='reg',
                    help='log file name')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=50,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=700,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--document_hidden_size', type=int, default=500,
                    help='hidden length that represents ducoment')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--aggregation', type=str, default='mean',
                    help='aggregation method, attention or mean')
parser.add_argument('--attention_width', type=int, default=10,
                    help='width of attention')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--bidirectional', action='store_true', help='LSTM: bidirectional or not')

parser.add_argument("--log_path", type=str, default="../logs")

parser.add_argument("--train_only", action='store_false', help="train or test")
parser.add_argument("--add_linear", action='store_true', help="add a linear layer or not")
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if not os.path.exists(args.log_path):
    os.makedirs(args.log_path)
logpath = os.path.join(args.log_path, args.model) + "_" + args.name + ".log"

logger = logger.Logger(logpath)

logger.Log(args)

###############################################################################
# Load data
###############################################################################

#data_set = data.load_data(args.data + '/topics_labeled_sample.xlsx')
data_set = data.load_data(args.data + '/training_set_rel3.tsv')
#print('loaded training')
logger.Log('loaded training')
data_size = len(data_set)
#print('data_size',data_size)
training_set = data_set[:int(data_size*0.8)]
dev_set = data_set[int(data_size*0.8):int(data_size*0.9)]
test_set = data_set[int(data_size*0.9):]

word_to_ix, index_to_word, vocab_size = data.build_dictionary([data_set])
#print('vocab size', vocab_size)
data.sentences_to_padded_index_sequences(word_to_ix, [training_set, dev_set, test_set])

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

def get_batch_with_id(batch):
    vectors = []
    texts = []
    labels = []
    ids = []
    essay_set_ids = []
    for dict in batch:
        vectors.append(dict["text_index_sequence"])
        texts.append(dict['text'])
        labels.append(dict["score"])
        ids.append(dict['id'])
        essay_set_ids.append(dict['essay_set'])
    return vectors, labels, texts, ids, essay_set_ids   

def training_loop(batch_size, num_epochs, model, loss_, optim, training_iter, dev_iter, train_eval_iter):
    step = 0
    epoch = 0
    total_batches = int(len(training_set) / batch_size)
    total_samples = total_batches * batch_size
    #hidden = model.init_hidden(batch_size)
    epoch_loss = 0
    best_dev_loss = None
    best_dev_kappa = None
    while epoch <= num_epochs:
        model.train()

        vectors, labels = get_batch(next(training_iter)) 
        #print(labels)
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        
        labels = torch.stack(labels).squeeze()
        if args.cuda:
            vectors = vectors.cuda()
            labels = labels.cuda() 
        vectors = Variable(vectors)
        labels = Variable(labels)

        hidden = model.init_hidden(batch_size)
        #hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden, _ = model(vectors, hidden)
        #logger.Log("output size {}".format(output.size()))
        #logger.Log("labels {}".format(labels.size()))
        #myloss = np.sum((output.data.cpu().numpy().reshape(batch_size) - labels.data.cpu().numpy())**2) / batch_size
        
        #logger.Log("hidden: " + str(hidden[0].size())) hidden: torch.Size([4, 100, 24])

        lossy = loss_(output, labels)
        epoch_loss += lossy.data[0] * batch_size

        lossy.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optim.step()


        if step % total_batches == 0:
            loss_train = evaluate(model, train_eval_iter)
            loss_dev = evaluate(model, dev_iter)
            kappa_dev = evaluate_kappa(model, dev_iter)
            logger.Log("Epoch %i; Step %i; Avg Loss %f; Train loss: %f; Dev loss: %f; Dev kappa: %f" 
                  %(epoch, step, epoch_loss/total_samples, loss_train, loss_dev, kappa_dev))
            epoch += 1
            epoch_loss = 0

            if not best_dev_loss or loss_dev < best_dev_loss:
                torch.save(model.state_dict(), args.save)
                #with open(args.save, 'wb') as f:
                #    torch.save(model, f)
                best_dev_loss = loss_dev
                best_dev_kappa = kappa_dev
                logger.Log("Save model with Best Dev loss: %f; Best Dev kappa: %f"%(best_dev_loss, best_dev_kappa))

        if step % 5 == 0:
            logger.Log("Epoch %i; Step %i; loss %f" %(epoch, step, lossy.data[0]))
        step += 1
    logger.Log("Best Dev loss: %f; Best Dev kappa: %f"%(best_dev_loss, best_dev_kappa))

# This function outputs the accuracy on the dataset, we will use it during training.
def evaluate(model, data_iter):
    model.eval()
    correct = 0
    total = 0
    evalloss = 0.0
    #hidden = model.init_hidden(args.batch_size)
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)
        labels = torch.stack(labels).squeeze()

        if args.cuda:
            vectors = vectors.cuda()
            labels = labels.cuda()
        vectors = Variable(vectors)

        hidden = model.init_hidden(args.batch_size)
        #hidden = repackage_hidden(hidden)
        output, hidden, _ = model(vectors, hidden)

        #print(F.mse_loss(output.data, labels).data)
        #print(loss(output.data, labels))
        evalloss += F.mse_loss(output.data, labels).data[0]
    return evalloss/len(data_iter)


# This function gives us the confusion matrix for all labels and the overall accuracy.
def evaluate_kappa(model, data_iter):
    model.eval()
    predicted_labels = []
    true_labels = []
    #hidden = model.init_hidden(args.batch_size)
    for i in range(len(data_iter)):
        vectors, labels = get_batch(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)

        if args.cuda:
            vectors = vectors.cuda()
        vectors = Variable(vectors)
        hidden = model.init_hidden(args.batch_size)
        #hidden = repackage_hidden(hidden)
        output, hidden, _ = model(vectors, hidden)

        predicted = [int(round(float(num))) for num in output.data.cpu().numpy()]
        predicted_labels.extend([round(float(num)) for num in output.data.cpu().numpy()])
        labels = [int(label[0]) for label in labels]
        true_labels.extend(labels)

    return cohen_kappa_score(true_labels, predicted_labels, weights = "quadratic")

def extract_features(model, data_iter):
    model.eval()
    predicted_labels = []
    true_labels = []
    #feature_hist = np.zeros((len(data_iter)*args.batch_size, args.nhid * args.nlayers))

    if args.add_linear:
        feature_hist = np.zeros((len(data_iter)*args.batch_size, args.document_hidden_size))
    else:
        feature_hist = np.zeros((len(data_iter)*args.batch_size, args.nhid * args.nlayers))
    texts_hist = []
    id_hist = []
    essay_set_hist = []
    features_dict = {}
    

    for i in range(len(data_iter)):
        vectors, labels, texts, ids, essay_set_ids = get_batch_with_id(data_iter[i])
        vectors = torch.stack(vectors).squeeze()
        vectors = vectors.transpose(1, 0)        

        if args.cuda:
            vectors = vectors.cuda()

        vectors = Variable(vectors)
        hidden = model.init_hidden(args.batch_size)
        #hidden = repackage_hidden(hidden)
        output, hidden, feature = model(vectors, hidden)
        #feature_ = feature.permute(1,0,2).view(args.batch_size, -1)
        #feature_hist.extend(feature.cpu().data.numpy())
        
        feature_hist[i*args.batch_size:(i+1)*args.batch_size, :] = feature.cpu().data.numpy()

        #feature_hist[i*args.batch_size:(i+1)*args.batch_size, :] = hidden[0].cpu().data.numpy()[2:, :, :].reshape(args.batch_size, -1)

        #predicted = [int(round(float(num))) for num in output.data.cpu().numpy()]
        predicted_label = [round(float(num)) for num in output.data.cpu().numpy()]
        predicted_labels.extend([round(float(num)) for num in output.data.cpu().numpy()])
        labels = [int(label[0]) for label in labels]
        true_labels.extend(labels)

        #for label, predicted_label, text, id_ in zip()
        texts_hist.extend(texts)
        id_hist.extend(ids)
        essay_set_hist.extend(essay_set_ids)

    return np.array(predicted_labels), np.array(true_labels), np.array(feature_hist), np.array(texts_hist), np.array(id_hist), np.array(essay_set_hist)

def cosine_similarity_(vec_one, vec_two):
    """
    Function that calculates the cosine similarity between two words
    """
    return float(cosine_similarity(np.array([vec_one,vec_two]))[0,1])


def similarity(true_labels, feature_hist, id_hist, essay_set_hist, essay_set = 3):
    index = np.where(essay_set_hist == essay_set)[0] 
    true_labels = true_labels[index]
    feature_hist = feature_hist[index]
    id_hist = id_hist[index]

    label_list = np.unique(true_labels)
    uni_lalel_size = len(label_list)
    matrix = np.zeros((len(label_list), len(label_list)))

    for k in range(uni_lalel_size):
        for l in range(k, uni_lalel_size):
            label1, label2 = k, l 

            index_score1 = np.where(true_labels == label1)[0]
            sub_feature_hist1 = feature_hist[index_score1]

            index_score2 = np.where(true_labels == label2)[0]
            sub_feature_hist2 = feature_hist[index_score2]
            diff = 0
            logger.Log("label{} len{}".format(label1, len(sub_feature_hist1)))
            logger.Log("label{} len{}".format(label2, len(sub_feature_hist2)))

            if label1 != label2:
                total = len(sub_feature_hist1) * len(sub_feature_hist2)
                for i in range(len(sub_feature_hist1)):
                    for j in range(len(sub_feature_hist2)):
                        diff+=cosine_similarity_(sub_feature_hist1[i], sub_feature_hist2[j])
                        #diff+=np.linalg.norm(sub_feature_hist1[i] - sub_feature_hist2[j])
            else:
                total = len(sub_feature_hist1) * (len(sub_feature_hist2)-1) / 2
                for i in range(len(sub_feature_hist1)):
                    for j in range(i+1, len(sub_feature_hist1)):
                        diff+=cosine_similarity_(sub_feature_hist1[i], sub_feature_hist1[j])
                        #diff+=np.linalg.norm(sub_feature_hist1[i] - sub_feature_hist2[j])

            diff/=total
            matrix[k][l] = diff

    confmx = """    score | 0 | 1 | 2 | 3
    -------------------------------------------------------
    0   |    {}    |    {}   |    {}    |     {}     
    1   |    {}    |    {}   |    {}    |     {}
    2   |    {}    |    {}   |    {}    |     {}
    3   |    {}    |    {}   |    {}    |     {}   """.format(\
        matrix[0][0],matrix[0][1],matrix[0][2],matrix[0][3],\
        matrix[1][0],matrix[1][1],matrix[1][2],matrix[1][3],\
        matrix[2][0],matrix[2][1],matrix[2][2],matrix[2][3],\
        matrix[3][0],matrix[3][1],matrix[3][2],matrix[3][3])

    return confmx

def cross_similarity(true_labels, feature_hist, id_hist, essay_set_hist, essay_set1 = 3, essay_set2 = 4):
    index30 = np.intersect1d(np.where(essay_set_hist == essay_set1)[0], np.where(true_labels == 0)[0])
    index33 = np.intersect1d(np.where(essay_set_hist == essay_set1)[0], np.where(true_labels == 3)[0])
    index40 = np.intersect1d(np.where(essay_set_hist == essay_set2)[0], np.where(true_labels == 0)[0])
    index43 = np.intersect1d(np.where(essay_set_hist == essay_set2)[0], np.where(true_labels == 3)[0])

    def cross(index_list1, index_list2, feature_hist):
        diff = 0
        m, n = len(index_list1), len(index_list2)
        for i in range(m):
            for j in range(n):
                diff+=cosine_similarity_(feature_hist[index_list1[i]], feature_hist[index_list2[j]])
                #diff+=np.linalg.norm(feature_hist[index_list1[i]] - feature_hist[index_list2[j]])
        return diff/(m*n)

    confmx = """score essay_set_3/essay_set_4 | 0 | 3
    ------------------------------------
    0   |    {}    |    {}    
    3   |    {}    |    {}   """.format(\
        cross(index30, index40, feature_hist),\
        cross(index30, index43, feature_hist),\
        cross(index33, index40, feature_hist),\
        cross(index33, index43, feature_hist))

    return confmx

def topK_similarity(true_labels, feature_hist, id_hist, essay_set_hist, K = 10, top = 3):
    index = np.random.randint(K, size=len(id_hist))
    size = len(true_labels)
    true_labels_sub = true_labels[index]
    feature_hist_sub = feature_hist[index]
    id_hist_sub = id_hist[index]
    essay_set_hist_sub = essay_set_hist[index]

    acc, acc1, acc2 = 0, 0, 0
    diff = []
    for i in index:
        diff_hist = []
        for j in range(size):
            if i != j:
                diff_hist.append(cosine_similarity_(feature_hist[i], feature_hist[j]))
        sorted_diff_hist = [i[0] for i in sorted(enumerate(diff_hist), key=lambda x:x[1])]
        smallest_diff_index = sorted_diff_hist[:top]
        for k in smallest_diff_index:
            if id_hist[k] == id_hist[i]:
                acc1 += 1
            if essay_set_hist[k] == essay_set_hist[i]:
                acc2 += 1
            if id_hist[k] == id_hist[i] and essay_set_hist[k] == essay_set_hist[i]:
                acc += 1
    return acc/K, acc1/K, acc2/K






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

###############################################################################
# Build the model
###############################################################################

# Build, initialize, and train model
rnn = regression.RNNModel(args, args.model, vocab_size, args.emsize, args.nhid, args.nlayers, dropout=args.dropout, bidirectional=args.bidirectional, pretrained_embedding=pretrained_embedding)
if args.cuda:
    rnn.cuda()

# Loss and Optimizer
loss = nn.MSELoss()
optimizer = torch.optim.Adadelta(rnn.parameters(), lr=args.lr)

# Train the model
training_iter = data_iter(training_set, args.batch_size)
train_eval_iter = eval_iter(training_set, args.batch_size)
dev_iter = eval_iter(dev_set, args.batch_size)
test_iter = eval_iter(test_set, args.batch_size)
#print('start training:')

if args.train_only:
    logger.Log('start training:')
    training_loop(args.batch_size, args.epochs, rnn, loss, optimizer, training_iter, dev_iter, train_eval_iter)

    logger.Log('finish training:')
    logger.Log('start loading best model:')
    #with open(args.save, 'rb') as f:
    #    rnn = torch.load(f)

    rnn.load_state_dict(torch.load(args.save))

    logger.Log('test on test set')
    loss_test = evaluate(rnn, test_iter)
    kappa_test = evaluate_kappa(rnn, test_iter)
    logger.Log("Test loss: %f; Best Test kappa: %f"%(loss_test, kappa_test))
        
    logger.Log('start calculating similarity:')

    total_data_iter = eval_iter(data_set, args.batch_size)

    predicted_labels, true_labels, feature_hist, texts_hist, id_hist, essay_set_hist = extract_features(rnn, total_data_iter)
    
    pickle.dump(feature_hist, open('feature_hist.p', 'wb'))
    pickle.dump(essay_set_hist, open('essay_set_hist.p', 'wb'))
    pickle.dump(true_labels, open('true_labels.p', 'wb'))    
    pickle.dump(predicted_labels, open('predicted_labels.p', 'wb')) 
    pickle.dump(id_hist, open('id_hist.p', 'wb'))   

    confmx = similarity(true_labels, feature_hist, id_hist, essay_set_hist)
    logger.Log(confmx)
    logger.Log('finish calculating single essay set similarity:')    

    logger.Log('start calculating cross essay set similarity:')
    confmx = cross_similarity(true_labels, feature_hist, id_hist, essay_set_hist, essay_set1 = 3, essay_set2 = 4)
    logger.Log(confmx)
    logger.Log('finish calculating cross essay set similarity:')

    #logger.Log('start calculating topK similarity:')
    #starttime = time.time()
    #acc, acc1, acc2 = topK_similarity(true_labels, feature_hist, id_hist, essay_set_hist)
    #logger.Log('time:' + str(time.time()-starttime))
    #logger.Log('finish calculating topK similarity with accuracy {}, {}, {}: '.format(acc, acc1, acc2))
 
else:
    logger.Log('start loading model:')
    
    #with open(args.save, 'rb') as f:
    #    rnn = torch.load(f)
    
    rnn.load_state_dict(torch.load(args.save))

    logger.Log('revaluate on dev set')
    loss_dev = evaluate(rnn, dev_iter)
    kappa_dev = evaluate_kappa(rnn, dev_iter)
    logger.Log("loaded model with dev loss: %f; dev kappa: %f"%(loss_dev, kappa_dev))

    logger.Log('start calculating similarity:')

    total_data_iter = eval_iter(data_set, args.batch_size)

    predicted_labels, true_labels, feature_hist, texts_hist, id_hist, essay_set_hist = extract_features(rnn, total_data_iter)

    
    confmx = similarity(true_labels, feature_hist, id_hist, essay_set_hist)
    logger.Log(confmx)
    logger.Log('finish calculating single essay set similarity:')    

    acc, acc1, acc2 = topK_similarity(true_labels, feature_hist, id_hist, essay_set_hist)
    logger.Log('finish calculating topK similarity with accuracy {}, {}, {}: '.format(acc, acc1, acc2))  
    #logger.Log(confmx)
"""
dev_full_iter = eval_iter(dev_set, args.batch_size)
print(evaluate_confusion(rnn, dev_full_iter))
with open(args.save, 'rb') as f:
    model = torch.load(f)

model.eval()
for i in range(len(test_iter)):
    for dict in test_iter[i]:
        vectors.append(dict["text_index_sequence"])
        labels.append(dict["score"])
return vectors, labels
loss_test = evaluate(model, test_iter)
kappa_test = evaluate_kappa(model, test_iter)
logger.Log("Test Loss %f; Test kappa: %f" %(loss_test, kappa_test))
"""
