import torch
import torch.nn as nn
import math
from models.MultiHeadAttention import MultiHeadAttention

class SelectedSharingLayer(nn.Module):
    """
    As defined in
        L Wu et al. 2019 EMNLP - Different Absorption from the Same Sharing:
                                                Sifted Multi-task Learning for Fake News Detection
    Inputs:
        Contextualized embedding of Stance/Severity detection solo through BERT
        Contextualized embedding of (Stance/Severity detection concatenated) through BERT
    Do:
        Do attention sharing
        Do gated sharing (similar to forget gate in LSTM)
        Concatenate the outputs and output it.
    """
    def __init__(self, merged_size, seq_len):
        """

        :param hidden_solo_size: [batch_size, sequence_length, hidden_solo_size]
        :param hidden_merged_size:
        :param batch_size:
        """

        super().__init__()

        # Gated Sharing parameters
        #self.weights = nn.Parameter(torch.Tensor(merged_size))
        #self.bias = nn.Parameter(torch.Tensor(seq_len))
        self.gate = nn.Linear(merged_size, merged_size)
        self.gate_sigmoid = torch.nn.Sigmoid()

        # Attention Mechanism
        self.num_heads = 10

        self.mh_attn = MultiHeadAttention(self.num_heads, merged_size, merged_size, merged_size)

    def forward(self, Efake, Hshared):
        """

        :param solo_unpooled_output: Either stance or severity, [batch_size, sequence_length, hidden_size]
        :param merged_unpooled_output: Both stance and severity, [batch_size, sequence_length, hidden_size]
        :return:
        """
        #[batch_size, sequence_length, hidden_size] x [hidden_size] -> [batch_size, sequence_length]
        #gated_values = self.gate_sigmoid(torch.matmul(merged_unpooled_output, self.weights) + self.bias)
        #gfake = sigmoid(Wfake * Hshared + bfake)
        gfake = self.gate_sigmoid(self.gate(Hshared))
        #Gfake = gfake (.) Hshared (element wise multiplication)
        Gfake = gfake * Hshared
        A, attn = self.mh_attn(Efake, Hshared, Hshared)

        # SSL = [G; |G-A|; G(.)A; A]
        SSL = torch.cat((Gfake, torch.abs(Gfake - A), Gfake*A, A), dim=-1)

        return SSL


