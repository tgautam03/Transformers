class Transformer(torch.nn.Module):
    def __init__(self, emb, heads, max_seq_length, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

        # Token Embedding
        self.token_embedding = torch.nn.Embedding(embedding_dim=emb, num_embeddings=vocab_size)
        # Positional Embedding
        self.pos_embedding = torch.nn.Embedding(embedding_dim=emb, num_embeddings=max_seq_length)

        # Self Attention
        self.attention = SelfAttention(emb, heads)

        # Layer Normalizations
        self.norm1 = torch.nn.LayerNorm(emb)
        self.norm2 = torch.nn.LayerNorm(emb)
        # FCN
        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(emb, 4*emb),
            torch.nn.ReLU(),
            torch.nn.Linear(4*emb, emb)
        )
        # Output
        self.toprobs = torch.nn.Linear(emb, 2)

    def forward(self, x):
        # Token Embedding
        tokens = self.token_embedding(x)
        b, t, e = tokens.shape
        
        # Positional Embedding
        positions = self.pos_embedding(torch.arange(t, device="cuda"))[None, :, :].expand(b, t, e)
        x = tokens + positions

        # Self Attention
        attented = self.attention(x)
        # FCN with normalizations
        x = self.norm1(attented + x)
        ff = self.fcn(x)
        x = self.norm2(ff + x)
        # Output 
        x = torch.mean(x, dim=1)
        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)