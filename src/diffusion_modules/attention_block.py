import torch
import torch.nn as nn
from typing import Optional

class BasicTransformerBlock(nn.Module):
    """
    Combines Self-Attention, Cross-Attention, and FeedForward using nn.MultiheadAttention.
    Operates on inputs of shape (B, C, H, W).
    Cross-Attention is applied conditionally based on context availability.
    """
    def __init__(self, dim: int, context_dim: int, n_head: int, dropout: float = 0.1):
        """
        Args:
            dim (int): Input dimension (channels)
            context_dim (int): Dimension of context embeddings (only used if context is provided)
            n_head (int): Number of attention heads
            dropout (float): Dropout rate
        """
        super().__init__()
        self.dim = dim
        # LayerNorms
        self.norm_self_attn = nn.LayerNorm(dim)
        self.norm_cross_attn = nn.LayerNorm(dim) # Norm before cross-attention
        self.norm_ff = nn.LayerNorm(dim)

        # Self-Attention Layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect input (B, N, C)
        )
        self.g_self = nn.Parameter(torch.zeros(1)) # Rezero parameter for self-attention

        # Cross-Attention Layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,        # Query dimension (from image features x)
            kdim=context_dim,     # Key dimension (from context)
            vdim=context_dim,     # Value dimension (from context)
            num_heads=n_head,
            dropout=dropout,
            batch_first=True # Expect query(B, N_img, C), key/value(B, N_ctx, C_ctx)
        )
        self.g_cross = nn.Parameter(torch.zeros(1)) # Rezero parameter for cross-attention

        # FeedForward Layer
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, C, H, W) - Image features
        # context: Optional[(B, seq_len_ctx, C_context)] - Text context embeddings or None
        batch_size, channels, height, width = x.shape
        n_tokens_img = height * width
        # Note: No residual = x here, residuals are added after each block

        # --- Reshape for Sequence Processing ---
        # (B, C, H, W) -> (B, C, N) -> (B, N, C)
        x_seq = x.view(batch_size, channels, n_tokens_img).transpose(1, 2)

        # --- Self-Attention ---
        x_norm = self.norm_self_attn(x_seq)
        self_attn_out, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, need_weights=False)
        x_seq = x_seq + self.g_self * self_attn_out # Add residual

        # --- Cross-Attention (Conditional) ---
        # Only perform cross-attention if context is provided
        if context is not None:
            x_norm = self.norm_cross_attn(x_seq)
            cross_attn_out, _ = self.cross_attn(query=x_norm, key=context, value=context, need_weights=False)
            x_seq = x_seq + self.g_cross * cross_attn_out # Add residual
        # If context is None, this block is skipped

        # --- FeedForward ---
        x_norm = self.norm_ff(x_seq)
        ff_out = self.ff(x_norm)
        x_seq = x_seq + ff_out # Add residual

        # --- Reshape back to Image Format ---
        # (B, N, C) -> (B, C, N) -> (B, C, H, W)
        out = x_seq.transpose(1, 2).view(batch_size, channels, height, width)

        return out # Return shape (B, C, H, W)
