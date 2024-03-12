"""
Various positional encodings for the transformer.
"""
#%%

import torch
from torch import nn

#%%
class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    num_embeddings:int= 50
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(self.num_embeddings, num_pos_feats)
        self.col_embed = nn.Embedding(self.num_embeddings, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:] #(h,w)
        i = torch.arange(w, device=x.device) #(w,)
        j = torch.arange(h, device=x.device) #(h,)
        x_emb = self.col_embed(i) # (w, num_post_feats) with w <= num_embeddings 
        y_emb = self.row_embed(j) # (h, num_post_feats) with h <= num_embeddings
        x_emb_unsqueezed = x_emb.unsqueeze(0) # (1, w, num_post_feats)
        y_emb_unsqueezed = y_emb.unsqueeze(1) # (h, 1, num_post_feats)
        x_emb_u_repeat = x_emb_unsqueezed.repeat(h,1,1) # (h,w,num_post_feats)
        y_emb_u_repeat = y_emb_unsqueezed.repeat(1,w,1) # (h,w,num_post_feats)

        positional_embedding = pos = torch.cat([
            x_emb_u_repeat, 
            y_emb_u_repeat
        ], dim=-1) # (h,w,2*num_post_feats)

        positional_embedding = positional_embedding.permute(2,0,1) # (2*num_post_feats, h,w)
        positional_embedding = positional_embedding.unsqueeze(0) # (1,2*num_post_feats,h,w)
        positional_embedding_batch = positional_embedding.repeat(x.shape[0], 1, 1, 1) # (B, 2*num_post_feats, h,w)
        return positional_embedding_batch


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    position_embedding = PositionEmbeddingLearned(N_steps)
    return position_embedding

#%%
x = torch.randn((32,15,15))
n_feats = 100 
pe = PositionEmbeddingLearned(n_feats)
pos_emb_batch = pe.forward(x)
print(pos_emb_batch.shape)


# COMMENTS # --- 
# Encoding always vary wrt to the embedding dimension 
"""
For each pixel (h,w), the value is different for every
dimension in 2*num_post_feats
"""

# Encoding horizontal location : 
"""
Horizontal location is defined by the first half of the cube. 
For the first half of 2*num_post_feats:
A row w : pos_emb_batch[0][0,0,:] varies
A column h : pos_emb_batch[0][0,:,0] does not vary

A pixel (h,w) has:
    - The same value as another pixel (h_i,w)
    - A different value as another pixel (h,w_i)
"""
#%%
# Encoding vertical location: 
"""
Vertical location is defined by the second half of the cube. 
For the second half of 2*num_post_feats:
A row w : pos_emb_batch[0][0,0,:] does not vary
A column h : pos_emb_batch[0][0,:,0] varies

A pixel (h,w) has:
    - The same value as another pixel (h,w_i)
    - A different value as another pixel (h_i,w)
"""


