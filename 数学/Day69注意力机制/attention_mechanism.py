import  torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self,  hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, query,keys):
        energy = self.W(keys)
        energy = torch.tanh(energy+query.unsqueeze(1))
        energy = self.v(energy).squeeze(-1)

        weights = F.softmax(energy, dim=1)

        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)

        return context, weights

batch_size = 2
seq_len = 5
hidden_dim = 10

query = torch.randn(batch_size, hidden_dim)
keys = torch.randn(batch_size, seq_len, hidden_dim)

attention = AdditiveAttention(hidden_dim)

context,weights = attention(query, keys)
print(context.shape)
print(weights.shape)
print("注意力机制:\n",weights)