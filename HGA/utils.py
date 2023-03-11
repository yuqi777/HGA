import csv
import io
import os
import shutil
import sys
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
import torch.nn as nn
# from konlpy.tag import Okt
from tqdm import tqdm
from sklearn import metrics
import torch
from loguru import logger
from os.path import join

def normalize_adj1(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = adj.cpu()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(abs(rowsum), -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    # print(d_mat_inv_sqrt.shape)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

from parse import parse_args
parser=parse_args()
def preprocess_adj1(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    x=[]
    for i in range(parser.batch_size):
        matrix = adj[i,:,:]
        adj_normalized = normalize_adj1(matrix + torch.eye(matrix.shape[1]).cuda())
        x.append(adj_normalized)
    # return sparse_to_tuple(adj_normalized)
    x=torch.tensor(x)
    return x.cuda()

def parsse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def accuracy(preds, labels):
    """Calculate accuracy"""
    correct_predict = (torch.argmax(preds, 1) == torch.argmax(labels, 1))
    acc = torch.mean(correct_predict.float()) 
    return acc

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed', 'y', 'tx_adj', 'tx_embed', 'ty', 'allx_adj', 'allx_embed', 'ally']
    objects = []
    if dataset_str == 'naver':
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='utf-8'))
                else:
                    objects.append(pkl.load(f))
    else:
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

    x_adj, x_embed, y, tx_adj, tx_embed, ty, allx_adj, allx_embed, ally = tuple(objects)   

    train_adj = []
    train_embed = []
    val_adj = []
    val_embed = []
    test_adj = []
    test_embed = []
    
    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        train_adj.append(adj)
        train_embed.append(embed)

    for i in range(len(y), len(ally)):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        val_adj.append(adj)
        val_embed.append(embed)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        test_adj.append(adj)
        test_embed.append(embed)

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    train_y = np.array(y)
    val_y = np.array(ally[len(y):len(ally)])
    test_y = np.array(ty)

    return train_adj, train_embed, train_y, val_adj, val_embed, val_y, test_adj, test_embed, test_y



def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])
    
    for i in range(features.shape[0]):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0] # padding for each epoch
        feature = np.pad(feature, ((0,pad),(0,0)), mode='constant')
        features[i] = feature
    
    return np.array(list(features))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    for i in range(adj.shape[0]):
        adj_normalized = normalize_adj(adj[i]) # no self-loop
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        mask[i,:adj[i].shape[0],:] = 1.
        adj[i] = adj_normalized

    return np.array(list(adj)), mask


def construct_feed_dict(features, support, mask, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def clean_str_naver(string):
    """
    Tokenization/string cleaning for the naver movie sentiment classification dataset
    """
    string = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", string) # remove except korean language
    string = re.sub('^ +', "", string) # blank => empty

    return string.strip()
class CSVWriter():
    def __init__(self, csv_path):
        super(CSVWriter, self).__init__()
        self.log_dir, self.csv_filename = os.path.split(csv_path)
        if not os.path.exists(csv_path):
            os.makedirs(self.log_dir, exist_ok=True)
            with open(csv_path, 'w'):
                pass
        self.csv_file = io.open(csv_path, 'a')
        self.fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.csv_file.flush()
    
    def write(self, **kwargs):
        filtered_kwargs = {k:v for k, v in kwargs.items() if k in self.fieldnames}
        self.writer.writerow(filtered_kwargs)
        self.csv_file.flush()

def getLogger(log_filename:str, log_dir:str, **kwargs):
    logger.remove()
    f_handler = logger.add(sink=join(log_dir, log_filename), level='INFO', format="{time:MM-DD HH:mm:ss}: {message}")
    c_handler = logger.add(sys.stderr, level='INFO', format='{message}')
    return logger
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def gather_idx(outs, indices,k):
    conv = nn.Conv1d(outs.shape[1], 1, 1).cuda().double()
    conv1 = nn.Conv1d(1, outs.shape[1], 1, 1).cuda().double()
    outs = conv(outs)
    outs = torch.gather(outs, 2, indices)
    outs = conv1(outs)
    return outs
def graph_pool( outs):
        k = 32
        conv = nn.Conv1d(outs.shape[1], 1, 1).cuda().double()
        scores = conv(outs)
        scores = torch.abs(scores)  #对分数求绝对值
        scores = torch.squeeze(scores, axis=2)  #将删除指定维度中值为1的变量
        scores = torch.sigmoid(scores)
        values, indices = torch.topk(scores, k)
        outs = gather_idx(outs, indices, k)
        outs = torch.mul(values, outs)
        return  outs