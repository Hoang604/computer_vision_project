import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from utils.network_comopents import SinusoidalPosEmb, Downsample, Upsample, DenseBlock, ResnetBlock, Residual, RRDB
from typing import Optional

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers, seq=False):
    """
    Make a layer of blocks
    :param block: block to be used
    :param n_layers: number of blocks
    :param seq: if True, return a Sequential layer
    :return: a Sequential layer or a ModuleList
    """
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    if seq:
        return nn.Sequential(*layers)
    else:
        return nn.ModuleList(layers)

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

        # Cross-Attention Layer (will be used conditionally)
        # We define it here, but only use it in forward if context is not None
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
    
class RRDBNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 out_channels=3, 
                 rrdb_in_channels=64, 
                 number_of_rrdb_blocks=8, 
                 growth_channels=32,
                 sr_scale=4):
        """
        Args:
            in_channels: input channels
            out_channels: output channels
            rrdb_in_channels: number of features - in features of RRDB
            number_of_rrdb_blocks: number of RRDB blocks
            growth_channels: growth channels
        """
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=rrdb_in_channels, gc=growth_channels)
        self.sr_scale = sr_scale
        # conv to extract from in_nc to nf feature, to feed into RRDB
        self.conv_first = nn.Conv2d(in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, number_of_rrdb_blocks)
        self.trunk_conv = nn.Conv2d(rrdb_in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(rrdb_in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(rrdb_in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        if self.sr_scale == 8:
            self.upconv3 = nn.Conv2d(rrdb_in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(rrdb_in_channels, rrdb_in_channels, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(rrdb_in_channels, out_channels, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, get_fea=False):
        feas = []
        x = (x + 1) / 2
        fea_first = fea = self.conv_first(x)  # shape [batch, nf, h, w]
        for l in self.RRDB_trunk:
            fea = l(fea)  # shape [batch, nf, h, w] 
            feas.append(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk  # shape [batch, nf, h, w]
        feas.append(fea)

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sr_scale == 8:
            fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        out = out * 2 - 1
        if get_fea:
            return out, feas
        else:
            return out


class Unet(nn.Module):
    def __init__(self, 
                 base_dim=64, 
                 out_dim=3, 
                 dim_mults=(1, 2, 4, 8), 
                 cond_dim=32,
                 rrdb_num_blocks=8,
                 sr_scale=4,
                 use_attention=False,
                 use_weight_norm=True,
                 weight_init=True):
        super().__init__()
        dims = [3, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        self.use_attention = use_attention

        num_raw_cond_channels = cond_dim * ((rrdb_num_blocks+ 1) // 3)
        self.cond_proj = nn.ConvTranspose2d(num_raw_cond_channels,
                                            base_dim, 
                                            sr_scale * 2, 
                                            sr_scale,
                                            sr_scale // 2)

        self.time_pos_emb = SinusoidalPosEmb(base_dim)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.Mish(),
            nn.Linear(base_dim * 4, base_dim)
        )

        self.context_spatial_pool = nn.AdaptiveAvgPool2d((20, 20))
        self.context_channel_proj = nn.Conv2d(num_raw_cond_channels, base_dim, kernel_size=1)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=base_dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=base_dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim, groups=groups)
        if use_attention:
            self.mid_attn = Residual(BasicTransformerBlock(
                dim=mid_dim,
                context_dim=base_dim,
                n_head=8,
                dropout=0.1))
        else:
            self.mid_attn = None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim, groups=groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=base_dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=base_dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            DenseBlock(base_dim, base_dim, groups=groups),
            nn.Conv2d(base_dim, out_dim, 1)
        )

        if use_weight_norm:
            self.apply_weight_norm()
        if weight_init:
            self.apply(initialize_weights)

    def apply_weight_norm(self):
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                # print(f"| Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x, time, condition):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        raw_cond_spatial = torch.cat(condition[2::3], 1) # Shape: (B, num_raw_cond_channels_for_attn_ctx, H_lr, W_lr)

        # ---- Prepare context for attention ----
        attention_context = None # Khởi tạo là None
        if self.use_attention and self.mid_attn is not None:            
            # 2. Spatial pooling
            pooled_cond_spatial = self.context_spatial_pool(raw_cond_spatial) 
            # Shape: (B, num_raw_cond_channels_for_attn_ctx, target_H_ctx, target_W_ctx)
            # Ex: (B, 96, 20, 20)
            
            # 3. Channel projection
            projected_channel_cond_spatial = self.context_channel_proj(pooled_cond_spatial) 
            # Shape: (B, dim, target_H_ctx, target_W_ctx)
            # Ex: (B, 64, 20, 20)

            # 4. Reshape for cross attention
            # Output shape: (B, target_H_ctx * target_W_ctx, dim)
            B, C_ctx, H_ctx, W_ctx = projected_channel_cond_spatial.shape
            attention_context = projected_channel_cond_spatial.view(B, C_ctx, H_ctx * W_ctx).transpose(1, 2)

        h = []
        projected_cond = self.cond_proj(raw_cond_spatial)
        for i, (resnet, resnet2, downsample) in enumerate(self.downs):
            x = resnet(x, t)
            x = resnet2(x, t)
            if i == 0:
                x = x + projected_cond
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        if self.use_attention and self.mid_attn is not None:
            x = self.mid_attn(x, context=attention_context)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)