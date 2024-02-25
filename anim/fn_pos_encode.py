import torch

pos_embedding = torch.nn.Embedding(embedding_dim=7, num_embeddings=max_seq_len)
positions = pos_embedding(torch.arange(5))[None, :, :].expand(1, 5, 7)