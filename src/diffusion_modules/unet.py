import torch
from torch import nn
from src.utils.network_comopents import SinusoidalPosEmb, Downsample, Upsample, DenseBlock, ResnetBlock, Residual
from src.diffusion_modules.attention_block import BasicTransformerBlock
from torch.nn import init


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


class Unet(nn.Module):
    def __init__(self, 
                 base_dim=64, 
                 out_dim=3, 
                 dim_mults=(1, 2, 4, 8), 
                 cond_dim=64,
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