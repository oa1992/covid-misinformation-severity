import torch
import torch.nn as nn
import math

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size, topic_embedding_size, dropout_prob=.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob

        # The query here is the topic embedding
        self.query = nn.Linear(topic_embedding_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, queries, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:-1])
        attention_mask = attention_mask.float()
        attention_mask = (1.0 - attention_mask) * -10000.0

        # [bsize, hidden_size] -> [bsize, 1, hidden_size]
        q = self.query(queries).view(-1, 1, self.hidden_size)
        # [bsize, seq_len, hidden_size]
        k = self.key(hidden_states)
        # [bsize, seq_len, hidden_size]
        v = self.value(hidden_states)

        # [bsize, 1, hidden_size] x [bsize, hidden_size, seq_len] -> [bsize, seq_len]
        attention_scores = torch.matmul(q, k.transpose(-1, -2)).view(-1, k.shape[1])
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        if attention_mask is not None:
            # apply the attention mask
            attention_scores = attention_scores + attention_mask

        # [bsize, seq_len]
        # normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)
        # [bsize, 1, seq_len] x [bsize, seq_len, hidden_size] -> [bsize, hidden_size]
        attention_probs = attention_probs.view(-1, 1, k.shape[1])

        context_layer = torch.matmul(attention_probs, v).view(-1, self.hidden_size)

        return context_layer