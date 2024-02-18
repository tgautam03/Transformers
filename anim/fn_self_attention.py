import torch
import torch.nn.functional as F

# Self Attention
def self_attention(X):
    # Weight Matrix
    W = torch.bmm(X, X.transpose(1, 2))
    W = W / (7**(1/2)) # Scaling for stability
    W = F.softmax(W, dim=-1)

    # Output Matrix
    y = torch.bmm(W, X)
    
    return y