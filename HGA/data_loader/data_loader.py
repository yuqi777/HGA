import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp

class TrainGenerator(Dataset):
    def __init__(self, args_config, adj, mask, emb, y):
        self.args_config = args_config
        self.adj = adj
        self.mask = mask
        self.emb = emb
        self.y = y
        self.n_nodes = len(self.adj)
    
    def __len__(self):
        return len(self.adj)
    
    def __getitem__(self, index):      
        train_id = np.random.randint(low=0, high=self.n_nodes, size=1)[0]    
        adj = self.adj[train_id]
        mask = self.mask[train_id]
        emb = self.emb[train_id]
        y = self.y[train_id]

        return adj, mask, emb, y

class TestGenerator(Dataset):
    def __init__(self, args_config, adj, mask, emb, y):
        self.args_config = args_config
        self.adj = adj
        self.mask = mask
        self.emb = emb
        self.y = y
        self.n_nodes = len(self.adj)

    def __len__(self):
        return len(self.adj)
    
    def __getitem__(self, index):
        test_id = np.random.randint(low=0, high=self.n_nodes, size=1)[0]
        adj = self.adj[test_id]
        mask = self.mask[test_id]
        emb = self.emb[test_id]
        y = self.y[test_id]

        return adj, mask, emb, y
class ValGenerator(Dataset):
    def __init__(self, args_config, adj, mask, emb, y):
        self.args_config = args_config
        self.adj = adj
        self.mask = mask
        self.emb = emb
        self.y = y
        self.n_nodes = len(self.adj)

    def __len__(self):
        return len(self.adj)
    
    def __getitem__(self, index):
        val_id = np.random.randint(low=0, high=self.n_nodes, size=1)[0]
        adj = self.adj[val_id]
        mask = self.mask[val_id]
        emb = self.emb[val_id]
        y = self.y[val_id]

        return adj, mask, emb, y