import torch
import torch.nn as nn
import torch.nn.functional as F
from parse import parse_args
from torch.nn.modules.module import Module
from utils import gather_idx,graph_pool
import numpy as np

args_config = parse_args()

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS= {}

def get_layer_uid(layer_name = ""):
    """assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, dropout, noise_shape):
    """Dropout for sparse tensors"""
    keep_porb = 1 - dropout
    random_tensor = (keep_porb + torch.rand(noise_shape)).floor()
    dropout_mask = torch.FloatTensor(random_tensor).type(torch.bool) #tf.cast
    i = x._indices()[:, dropout_mask]
    v = x._values()[dropout_mask] * (1.0/keep_porb)
    
    return torch.sparse.FloatTensor(i,v)


def sparse_dense_matmul_batch(sp_a, b):
    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = sp_a[i][:sp_a.shape[1]][:sp_a.shape[2]]
        sparse_slice = torch.reshape(sparse_slice, (sp_a.shape[1], sp_a.shape[2]))
        mult_slice = torch.matmul(sparse_slice, dense_slice)

        return mult_slice
    
    elems = (torch.range(0, sp_a.shape[0], step=1, dtype=torch.int64), b)
    map_tensor = map_function(elems).type(torch.float64)

    return map_tensor


def dot(x, y, sparse=False):
    
    if sparse:
        res = sparse_dense_matmul_batch(x, y)
    else:
        res = torch.matmul(x, y)

    return res.cuda()


def gru_unit(support, x, weights, biases, act, mask, dropout, sparse_inputs=False):
    #x代表前一层的隐藏层状态，在刚开始初始化的时候是文本的原始嵌入向量
    """GRU unit with 3D tensor inputs."""
    # message passing
    support = F.dropout(support, dropout)
    a = torch.matmul(support, x)

    # update gate        
    z0 = dot(a, weights[1], sparse_inputs) + biases[1]
    z1 = dot(x, weights[2], sparse_inputs) + biases[2] 
    z = torch.sigmoid(z0 + z1)
    
    # reset gate
    r0 = dot(a, weights[3], sparse_inputs) + biases[3]
    r1 = dot(x, weights[4], sparse_inputs) + biases[4]
    r = torch.sigmoid(r0 + r1)

    # update embeddings    
    h0 = dot(a, weights[5], sparse_inputs) + biases[5]
    h1 = dot(r*x, weights[6], sparse_inputs) + biases[6]
    h = act(mask * (h0 + h1))
    
    return h*z + x*(1-z)

class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch, bidirectional=True, batch_first=True):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional, batch_first).double()
        self.h_0 = torch.randn(num_layers, batch, hidden_size).double()
        self.c_0 = torch.randn(num_layers, batch, hidden_size).double()
        if torch.cuda.is_available:
            self.h_0 , self.c_0 = self.h_0.cuda() , self.c_0.cuda()

    def forward(self, x):
        out, (h_n, h_c) = self.rnn(x, (self.h_0, self.c_0))
        # out = torch.max(out,dim=1)[0]
        # print(out.shape)
        return out


class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv1d(self.hidden_size, 32, kernel_size=(1, 4)).cuda().double()
        self.conv2 = nn.Conv1d(self.hidden_size, 32, kernel_size=(1, 5)).cuda().double()
        
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(2)   # B * C * H * W

        out1 = F.adaptive_avg_pool2d(F.relu(self.conv1(x)), 1)
        out2 = F.adaptive_avg_pool2d(F.relu(self.conv2(x)), 1)
        out = torch.cat([out1, out2], dim=1).squeeze()
        # print(out.shape)
        # if self.is_fc:
        #     out = self.fc(out)
        return out


    
class Dense_layer(Module):
    """Dense layer"""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, 
                act = F.relu, bias_use=False, featureless=False, num_features_non=0.):
        
        super(Dense_layer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_non = num_features_non
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias_use = bias_use
        self.weight = nn.parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        
        nn.init.xavier_uniform_(self.weight)
        if self.biasless:
            self.bias = nn.parameter(torch.FloatTensor(self.output_dim))
    
    def forward(self, inputs):
        
        if self.sparse_inputs:
            x = sparse_dropout(inputs, self.dropout, self.num_features_non)
        else:
            x = F.dropout(x, self.dropout)        
        # transform
        output = dot(x, self.weight, sparse=self.sparse_inputs)
        # bias 
        if self.bias:
            output += self.bias
        
        return self.act(output)


class Graph_layer(Module):
    """Graph layer."""
    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, 
                act = F.relu, bias_use=False, featureless=False, steps=2):
        
        super(Graph_layer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features_non = 0.        
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias_use = bias_use
        self.steps = steps

        self.weights = []
        self.weight_encode = nn.Parameter(torch.DoubleTensor(self.input_dim, self.output_dim))
        self.weights.append(self.weight_encode)
        self.weight_z0 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_z0)
        self.weight_z1 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_z1)
        self.weight_r0 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_r0)
        self.weight_r1 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_r1)
        self.weight_h0 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_h0)
        self.weight_h1 = nn.Parameter(torch.DoubleTensor(self.output_dim, self.output_dim))
        self.weights.append(self.weight_h1)

        self.biases = []
        self.bias_encode = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_encode)
        self.bias_z0 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_z0)
        self.bias_z1 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_z1)
        self.bias_r0 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_r0)
        self.bias_r1 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_r1)
        self.bias_h0 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_h0)
        self.bias_h1 = nn.Parameter(torch.zeros(self.output_dim))
        self.biases.append(self.bias_h1)

        self.reset_parameters()
    
    def reset_parameters(self):
        
        nn.init.xavier_uniform_(self.weight_encode)
        nn.init.xavier_uniform_(self.weight_z0)
        nn.init.xavier_uniform_(self.weight_z1)
        nn.init.xavier_uniform_(self.weight_r0)
        nn.init.xavier_uniform_(self.weight_r1)
        nn.init.xavier_uniform_(self.weight_h0)
        nn.init.xavier_uniform_(self.weight_h1)        
    
    def forward(self, inputs, support, mask):

        self.support = support
        self.mask = mask

        if self.sparse_inputs:
            x = sparse_dropout(inputs, self.dropout, self.num_features_non)
        else:
            x = F.dropout(inputs, self.dropout)  

        # encode inputs
        x = dot(x, self.weight_encode, sparse=self.sparse_inputs) + self.bias_encode
        output = self.mask * self.act(x)

        # convolve
        for _ in range(self.steps):
            output = gru_unit(self.support, output, self.weights, self.biases, self.act, 
                                self.mask, self.dropout, self.sparse_inputs)
        
        return output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,kernel_size=1,padding=0, bias=False).cuda(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,kernel_size=1,padding=0, bias=False).cuda(),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.type(torch.FloatTensor).cuda()
        x = x.unsqueeze(-1)
        # print('x x',x.shape)
        x = x.permute(0,2,1,3)
        y = self.avg_pool(x) #对应Squeeze操作
        y = self.fc(y) #对应Excitation操作
        
        new_y = x * y.expand_as(x)
        new_y = new_y.squeeze()
        new_y = new_y.permute(0,2,1) # [32,50,900]
        return new_y

class Readout_layer(Module):

    def __init__(self, input_dim, output_dim, dropout=0., sparse_inputs=False, 
                act = F.relu, bias_use=False):
        
        super(Readout_layer, self).__init__()
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.sparse_inputs = sparse_inputs        
        self.bias_use = bias_use 
        self.Relu=nn.ReLU()       
        self.seblock=SELayer(args_config.hidden)
        # self.rnn=RNNEncoder(input_size=self.input_dim, hidden_size=args_config.hidden, num_layers=2, batch=args_config.batch_size )

        self.weights = []
        self.weight_att = nn.Parameter(torch.DoubleTensor(self.input_dim, 1))  
        self.weights.append(self.weight_att)      
        self.weight_emb = nn.Parameter(torch.DoubleTensor(self.input_dim, self.input_dim))    
        self.weights.append(self.weight_emb)   
        self.weight_mlp = nn.Parameter(torch.DoubleTensor(self.input_dim, self.output_dim))
        self.weights.append(self.weight_mlp)
        
        self.biases = []
        self.bias_att = nn.Parameter(torch.zeros(1))  
        self.biases.append(self.bias_att)      
        self.bias_emb = nn.Parameter(torch.zeros(self.input_dim))      
        self.biases.append(self.bias_emb)  
        self.bias_mlp = nn.Parameter(torch.zeros(self.output_dim))    
        self.biases.append(self.bias_mlp)    

        self.reset_parameters()
    
    def reset_parameters(self):
        
        nn.init.xavier_uniform_(self.weight_att)
        nn.init.xavier_uniform_(self.weight_emb)
        nn.init.xavier_uniform_(self.weight_mlp)

    def forward(self, inputs, mask):

        # out = F.dropout(out, self.dropout)
        
        self.mask = mask       
        x = inputs

        # soft attention
        att = torch.sigmoid(dot(x, self.weight_att) + self.bias_att)
        emb = self.act(dot(x, self.weight_emb) + self.bias_emb)
        N = torch.sum(self.mask, dim=1)
        M = (self.mask - 1) * 1e-9

        # graph summation
        g = self.mask * att * emb    #    [32, 119, 64]
        seg=self.seblock(g)
        # save=seg.cpu().detach().numpy()
        # xx=x.cpu().detach().numpy()
        # b = save.reshape(-1,)
        # np.savetxt("test_data.txt", b)
        # np.savetxt("test_input.txt", xx)
        g = (torch.sum(g, dim=1)) + (torch.max(g+M, dim=1)[0])
        g=g+torch.max(seg,dim=1)[0]

        g = F.dropout(g, self.dropout)
        # print(g.shape,self.weight_mlp.shape)
        # classification
        output = torch.matmul(g, self.weight_mlp) + self.bias_mlp
        # print(output.shape)
        return output   