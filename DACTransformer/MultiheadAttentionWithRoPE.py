import torch
import torch.nn as nn
import torch.nn.functional as F




class MultiheadAttentionWithRoPE(nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, rotary_positional_embedding, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.rotary_positional_embedding = rotary_positional_embedding

    def forward(self, query, key, value, **kwargs):
        # Manually compute Q, K, V projections
        q = F.linear(query, self.in_proj_weight[:self.embed_dim], self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim:2*self.embed_dim], self.in_proj_bias[self.embed_dim:2*self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2*self.embed_dim:], self.in_proj_bias[2*self.embed_dim:])

        # Apply Rotary Positional Encoding (RoPE) to Q and K
        q = self.rotary_positional_embedding(q)
        k = self.rotary_positional_embedding(k)

        # Use the parent's forward method for attention computation
        return super().forward(q, k, v, **kwargs)