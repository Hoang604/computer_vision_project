import torch
from torch import nn
from src.utils.network_comopents import SinusoidalPosEmb, Downsample, Upsample, DenseBlock, ResnetBlock, Residual
from src.diffusion_modules.attention_block import BasicTransformerBlock
from torch.nn import init
from typing import List, Optional, Tuple, Union


def initialize_weights(net_l, scale=0.1):
    """Initialize network weights with Kaiming normal initialization."""
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


class AdditiveConditioningBlock(nn.Module):
    """
    Additive conditioning mechanism that projects conditioning features 
    to match the target dimension and adds them directly.
    """
    def __init__(self, cond_channels: int, target_channels: int, scale_factor: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Project condition channels to target channels
        self.cond_proj = nn.Sequential(
            nn.Conv2d(cond_channels, target_channels, kernel_size=1),
            nn.GroupNorm(min(8, target_channels), target_channels),
            nn.SiLU()
        )
        
        # Optional upsampling if needed
        if scale_factor > 1:
            self.upsample = nn.ConvTranspose2d(
                target_channels, target_channels, 
                kernel_size=scale_factor*2, stride=scale_factor, 
                padding=scale_factor//2
            )
        else:
            self.upsample = nn.Identity()
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, C, H, W)
            condition: Conditioning features (B, cond_C, cond_H, cond_W)
        Returns:
            x + projected_condition
        """
        cond_proj = self.cond_proj(condition)
        cond_proj = self.upsample(cond_proj)
        
        # Resize to match x if necessary
        if cond_proj.shape[-2:] != x.shape[-2:]:
            cond_proj = nn.functional.interpolate(
                cond_proj, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
        
        return x + cond_proj


class ConcatenationConditioningBlock(nn.Module):
    """
    Concatenation conditioning mechanism that concatenates conditioning features
    with input features along the channel dimension.
    """
    def __init__(self, input_channels: int, cond_channels: int, output_channels: int, scale_factor: int = 1):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Project condition to appropriate resolution
        if scale_factor > 1:
            self.cond_upsample = nn.ConvTranspose2d(
                cond_channels, cond_channels, 
                kernel_size=scale_factor*2, stride=scale_factor, 
                padding=scale_factor//2
            )
        else:
            self.cond_upsample = nn.Identity()
        
        # Projection after concatenation
        concat_channels = input_channels + cond_channels
        self.output_proj = nn.Sequential(
            nn.Conv2d(concat_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU()
        )
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, in_C, H, W)
            condition: Conditioning features (B, cond_C, cond_H, cond_W)
        Returns:
            projected concatenated features
        """
        cond_upsampled = self.cond_upsample(condition)
        
        # Resize to match x if necessary
        if cond_upsampled.shape[-2:] != x.shape[-2:]:
            cond_upsampled = nn.functional.interpolate(
                cond_upsampled, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
        
        # Concatenate along channel dimension
        concat_features = torch.cat([x, cond_upsampled], dim=1)
        
        return self.output_proj(concat_features)


class UnetImproved(nn.Module):
    """
    Improved U-Net with multi-level attention and alternative conditioning mechanisms.
    
    Features:
    - Multi-level attention at different resolutions
    - Multiple conditioning strategies (cross-attention, additive, concatenation)
    - Flexible attention placement
    """
    
    def __init__(self, 
                 base_dim=64, 
                 out_dim=3, 
                 dim_mults=(1, 2, 4, 8), 
                 cond_dim=64,
                 rrdb_num_blocks=8,
                 sr_scale=4,
                 use_attention=True,
                 attention_levels: List[int] = [2, 3],  # Which levels to apply attention (0-indexed)
                 conditioning_strategy: str = "mixed",  # "cross_attention", "additive", "concatenation", "mixed"
                 use_weight_norm=True,
                 weight_init=True,
                 attention_heads=8):
        super().__init__()
        
        self.base_dim = base_dim
        self.out_dim = out_dim
        self.dim_mults = dim_mults
        self.rrdb_num_blocks = rrdb_num_blocks
        self.cond_dim = cond_dim
        self.sr_scale = sr_scale
        self.use_attention = use_attention
        self.attention_levels = attention_levels
        self.conditioning_strategy = conditioning_strategy
        self.use_weight_norm = use_weight_norm
        self.weight_init = weight_init
        
        dims = [3, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        groups = 0
        
        num_raw_cond_channels = cond_dim * ((rrdb_num_blocks + 1) // 3)
        
        # Original conditioning projection for cross-attention
        if conditioning_strategy in ["cross_attention", "mixed"]:
            self.cond_proj = nn.ConvTranspose2d(
                num_raw_cond_channels, base_dim, 
                sr_scale * 2, sr_scale, sr_scale // 2
            )
            
            self.context_spatial_pool = nn.AdaptiveAvgPool2d((20, 20))
            self.context_channel_proj = nn.Conv2d(num_raw_cond_channels, base_dim, kernel_size=1)
        
        # Time embedding
        self.time_pos_emb = SinusoidalPosEmb(base_dim)
        self.mlp = nn.Sequential(
            nn.Linear(base_dim, base_dim * 4),
            nn.Mish(),
            nn.Linear(base_dim * 4, base_dim)
        )
        
        # Down path
        self.downs = nn.ModuleList([])
        self.down_attentions = nn.ModuleList([])
        self.down_conditioning_blocks = nn.ModuleList([])
        
        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            # Main down blocks
            down_blocks = nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=base_dim, groups=groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=base_dim, groups=groups),
                Downsample(dim_out) if not is_last else nn.Identity()
            ])
            self.downs.append(down_blocks)
            
            # Attention at specific levels
            if use_attention and ind in attention_levels:
                attention_block = Residual(BasicTransformerBlock(
                    dim=dim_out,
                    context_dim=base_dim,
                    n_head=attention_heads,
                    dropout=0.1
                ))
                self.down_attentions.append(attention_block)
            else:
                self.down_attentions.append(nn.Identity())
            
            # Alternative conditioning blocks
            if conditioning_strategy == "additive":
                scale_factor = sr_scale // (2 ** ind) if sr_scale // (2 ** ind) >= 1 else 1
                cond_block = AdditiveConditioningBlock(
                    cond_channels=num_raw_cond_channels,
                    target_channels=dim_out,
                    scale_factor=scale_factor
                )
                self.down_conditioning_blocks.append(cond_block)
            elif conditioning_strategy == "concatenation":
                scale_factor = sr_scale // (2 ** ind) if sr_scale // (2 ** ind) >= 1 else 1
                cond_block = ConcatenationConditioningBlock(
                    input_channels=dim_out,
                    cond_channels=num_raw_cond_channels,
                    output_channels=dim_out,
                    scale_factor=scale_factor
                )
                self.down_conditioning_blocks.append(cond_block)
            elif conditioning_strategy == "mixed":
                # Mix additive and concatenation at different levels
                if ind % 2 == 0:  # Even levels use additive
                    scale_factor = sr_scale // (2 ** ind) if sr_scale // (2 ** ind) >= 1 else 1
                    cond_block = AdditiveConditioningBlock(
                        cond_channels=num_raw_cond_channels,
                        target_channels=dim_out,
                        scale_factor=scale_factor
                    )
                else:  # Odd levels use concatenation
                    scale_factor = sr_scale // (2 ** ind) if sr_scale // (2 ** ind) >= 1 else 1
                    cond_block = ConcatenationConditioningBlock(
                        input_channels=dim_out,
                        cond_channels=num_raw_cond_channels,
                        output_channels=dim_out,
                        scale_factor=scale_factor
                    )
                self.down_conditioning_blocks.append(cond_block)
            else:
                self.down_conditioning_blocks.append(nn.Identity())
        
        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim, groups=groups)
        
        if use_attention:
            self.mid_attn = Residual(BasicTransformerBlock(
                dim=mid_dim,
                context_dim=base_dim,
                n_head=attention_heads,
                dropout=0.1
            ))
        else:
            self.mid_attn = None
            
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=base_dim, groups=groups)
        
        # Up path
        self.ups = nn.ModuleList([])
        self.up_attentions = nn.ModuleList([])
        self.up_conditioning_blocks = nn.ModuleList([])
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            level_ind = num_resolutions - 2 - ind  # Map back to down level index
            
            # Main up blocks
            up_blocks = nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=base_dim, groups=groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=base_dim, groups=groups),
                Upsample(dim_in) if not is_last else nn.Identity()
            ])
            self.ups.append(up_blocks)
            
            # Attention at specific levels (mirror down path)
            if use_attention and level_ind in attention_levels:
                attention_block = Residual(BasicTransformerBlock(
                    dim=dim_in,
                    context_dim=base_dim,
                    n_head=attention_heads,
                    dropout=0.1
                ))
                self.up_attentions.append(attention_block)
            else:
                self.up_attentions.append(nn.Identity())
            
            # Alternative conditioning blocks (mirror down path)
            if conditioning_strategy == "additive":
                scale_factor = sr_scale // (2 ** level_ind) if sr_scale // (2 ** level_ind) >= 1 else 1
                cond_block = AdditiveConditioningBlock(
                    cond_channels=num_raw_cond_channels,
                    target_channels=dim_in,
                    scale_factor=scale_factor
                )
                self.up_conditioning_blocks.append(cond_block)
            elif conditioning_strategy == "concatenation":
                scale_factor = sr_scale // (2 ** level_ind) if sr_scale // (2 ** level_ind) >= 1 else 1
                cond_block = ConcatenationConditioningBlock(
                    input_channels=dim_in,
                    cond_channels=num_raw_cond_channels,
                    output_channels=dim_in,
                    scale_factor=scale_factor
                )
                self.up_conditioning_blocks.append(cond_block)
            elif conditioning_strategy == "mixed":
                if level_ind % 2 == 0:  # Even levels use additive
                    scale_factor = sr_scale // (2 ** level_ind) if sr_scale // (2 ** level_ind) >= 1 else 1
                    cond_block = AdditiveConditioningBlock(
                        cond_channels=num_raw_cond_channels,
                        target_channels=dim_in,
                        scale_factor=scale_factor
                    )
                else:  # Odd levels use concatenation
                    scale_factor = sr_scale // (2 ** level_ind) if sr_scale // (2 ** level_ind) >= 1 else 1
                    cond_block = ConcatenationConditioningBlock(
                        input_channels=dim_in,
                        cond_channels=num_raw_cond_channels,
                        output_channels=dim_in,
                        scale_factor=scale_factor
                    )
                self.up_conditioning_blocks.append(cond_block)
            else:
                self.up_conditioning_blocks.append(nn.Identity())
        
        # Final output
        self.final_conv = nn.Sequential(
            DenseBlock(base_dim, base_dim, groups=groups),
            nn.Conv2d(base_dim, out_dim, 1)
        )
        
        # Apply weight norm and initialization
        if use_weight_norm:
            self.apply_weight_norm()
        if weight_init:
            self.apply(initialize_weights)
    
    def apply_weight_norm(self):
        """Apply weight normalization to conv layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
        
        self.apply(_apply_weight_norm)
    
    def prepare_attention_context(self, raw_cond_spatial: torch.Tensor) -> Optional[torch.Tensor]:
        """Prepare context for cross-attention."""
        if not self.use_attention or self.conditioning_strategy == "additive" or self.conditioning_strategy == "concatenation":
            return None
        
        # Spatial pooling
        pooled_cond_spatial = self.context_spatial_pool(raw_cond_spatial)
        
        # Channel projection
        projected_channel_cond_spatial = self.context_channel_proj(pooled_cond_spatial)
        
        # Reshape for cross attention
        B, C_ctx, H_ctx, W_ctx = projected_channel_cond_spatial.shape
        attention_context = projected_channel_cond_spatial.view(B, C_ctx, H_ctx * W_ctx).transpose(1, 2)
        
        return attention_context
    
    def forward(self, x, time, condition):
        """
        Forward pass of the improved U-Net.
        
        Args:
            x: Input tensor (B, C, H, W)
            time: Time step tensor (B,)
            condition: List of conditioning features from RRDBNet
        
        Returns:
            Output tensor (B, out_dim, H, W)
        """
        # Time embedding
        t = self.time_pos_emb(time)
        t = self.mlp(t)
        
        # Prepare conditioning features
        raw_cond_spatial = torch.cat(condition[2::3], 1)  # Shape: (B, num_raw_cond_channels, H_lr, W_lr)
        
        # Prepare context for attention
        attention_context = self.prepare_attention_context(raw_cond_spatial)
        
        # Original conditioning projection for first layer (if using cross-attention or mixed)
        if hasattr(self, 'cond_proj'):
            projected_cond = self.cond_proj(raw_cond_spatial)
        else:
            projected_cond = None
        
        # Down path
        h = []
        for i, down_block in enumerate(self.downs):
            attn_block = self.down_attentions[i]
            cond_block = self.down_conditioning_blocks[i]
            
            # Extract individual layers from ModuleList
            resnet1, resnet2, downsample = down_block
            
            x = resnet1(x, t)
            x = resnet2(x, t)
            
            # Apply original conditioning at first layer
            if i == 0 and projected_cond is not None:
                x = x + projected_cond
            
            # Apply alternative conditioning
            if not isinstance(cond_block, nn.Identity):
                x = cond_block(x, raw_cond_spatial)
            
            # Apply attention if available
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context=attention_context)
            
            h.append(x)
            x = downsample(x)
        
        # Middle blocks
        x = self.mid_block1(x, t)
        if self.mid_attn is not None:
            x = self.mid_attn(x, context=attention_context)
        x = self.mid_block2(x, t)
        
        # Up path
        for i, up_block in enumerate(self.ups):
            attn_block = self.up_attentions[i]
            cond_block = self.up_conditioning_blocks[i]
            
            # Extract individual layers from ModuleList
            resnet1, resnet2, upsample = up_block
            
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            
            # Apply alternative conditioning
            if not isinstance(cond_block, nn.Identity):
                x = cond_block(x, raw_cond_spatial)
            
            # Apply attention if available
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context=attention_context)
            
            x = upsample(x)
        
        return self.final_conv(x)
    
    def make_generation_fast_(self):
        """Remove weight normalization for faster inference."""
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return
        
        self.apply(remove_weight_norm)
