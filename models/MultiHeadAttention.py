import torch
import torch.nn as nn
import torch.nn.functional as f
from models.ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, input_dim, dim_keys, dim_values, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.dim_keys = dim_keys
        self.dim_values = dim_values

        ### CHANGE TO *2/3 for test[9], /2 for test[:8], reg for test[10]

        self.elongate_queries = nn.Linear(int(input_dim/2), input_dim)
        #self.elongate_queries = nn.Linear(int(input_dim*2/3), input_dim)
        self.weight_queries = nn.Linear(input_dim, num_heads*dim_keys, bias=False)
        self.weight_keys = nn.Linear(input_dim, num_heads*dim_keys, bias=False)
        self.weight_values = nn.Linear(input_dim, num_heads*dim_keys, bias=False)
        self.fc = nn.Linear(num_heads * dim_values, input_dim, bias=False)
        #self.queries_reduce = nn.Linear(input_dim, int(input_dim/2), bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_keys ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

    def forward(self, queries, keys, values, mask=None):
        dim_keys, dim_values, num_heads = self.dim_keys, self.dim_values, self.num_heads
        batch_size, seq_len_q, seq_len_k, seq_len_v = queries.size(0), queries.size(1), keys.size(1), values.size(1)

        queries = self.elongate_queries(queries)
        residual = queries

        # Pass through the pre-attention projection: batch_size x seq_len_q x num_heads*dim_val]
        # separate different heads
        queries = self.weight_queries(queries).view(batch_size, seq_len_q, num_heads, dim_keys)
        keys    = self.weight_keys(keys).view(batch_size, seq_len_k, num_heads, dim_keys)
        values  = self.weight_values(values).view(batch_size, seq_len_v, num_heads, dim_values)

        # Transpose for attention dot product: b x n x lq x dv
        queries, keys, values = queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # for head axis broadcasting

        queries, attention = self.attention(queries, keys, values, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        queries = queries.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        queries = self.dropout(self.fc(queries))
        queries += residual

        queries = self.layer_norm(queries)
        return queries, attention

