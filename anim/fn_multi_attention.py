import torch
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        assert k % heads == 0
        self.k, self.heads = k, heads

        # Three k x k matrix multiplications to get queries, keys and values
        self.to_queries = torch.nn.Linear(k, k)
        self.to_keys = torch.nn.Linear(k, k)
        self.to_values = torch.nn.Linear(k, k)

        # One last Linear layer at the end with k x k matrix multiplication
        self.unify = torch.nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.shape # Num Batches, Batch Size, Sequence Length
        h = self.heads

        # Computing queries, keys and values
        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        # Slicing out the heads
        queries = queries.view(b, t, h, k/h)
        keys = keys.view(b, t, h, k/h)
        values = values.view(b, t, h, k/h)

        # Folding heads into batch dims (Remember head computations can run in parallel)
        queries = queries.transpose(1, 2).reshape(b*h, t, k/h)
        keys = keys.transpose(1, 2).reshape(b*h, t, k/h)
        values = values.transpose(1, 2).reshape(b*h, t, k/h)

        # Here comes Self Attention...
        W = torch.bmm(queries, keys.transpose(1,2)) # Computing Weights
        W = W / (k**(1/2)) # Scaling for stability
        W = F.softmax(W, dim=2) # Row-wise Softmax
        y = torch.bmm(W, values).view(b, h, t, k/h) # Computing y
        y = y.transpose(1, 2).reshape(b, t, k) # Concatenating heads

        # Final Linear NN Layer
        return self.unify(y)