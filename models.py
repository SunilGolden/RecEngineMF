import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)
    
    def forward(self, users, items):
        U = self.user_emb(users)
        V = self.item_emb(items)
        b_u = self.user_bias(users).squeeze()
        b_v = self.item_bias(items).squeeze()
        
        return (U * V).sum(dim=1) + b_u + b_v