import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Module to generate sinusoidal position embeddings for time steps t.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time (torch.Tensor): Tensor containing time steps, shape (batch_size,).

        Returns:
            torch.Tensor: Positional embedding tensor, shape (batch_size, dim).
        """
        device = time.device
        half_dim = self.dim // 2
        # Correct calculation for frequencies: 1 / (10000^(2i/dim))
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000.0) / half_dim))
        # Unsqueeze time for broadcasting: time shape (B, 1), embeddings shape (half_dim,) -> (B, half_dim)
        embeddings = time.unsqueeze(1) * embeddings.unsqueeze(0)
        # Concatenate sin and cos: shape (B, dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # Handle odd dimensions if necessary
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g
    
class DenseBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        if groups == 0:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.Mish()
            )
        else:
            self.block = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim_out, 3),
                nn.GroupNorm(groups, dim_out),
                nn.Mish()
            )

    def forward(self, x):
        return self.block(x)
    

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=0, groups=8):
        super().__init__()
        if time_emb_dim > 0:
            self.mlp = nn.Sequential(
                nn.Mish(),
                nn.Linear(time_emb_dim, dim_out * 2)
            )

        self.block1 = DenseBlock(dim, dim_out, groups=groups)
        self.block2 = DenseBlock(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond=None):
        h = self.block1(x)
        if time_emb is not None and hasattr(self, 'mlp'):
            # self.mlp(time_emb) has shape (B, dim_out * 2)
            # time_proj has shape (B, dim_out * 2, 1, 1)
            time_proj = self.mlp(time_emb).unsqueeze(-1).unsqueeze(-1) 
            
            # scale and shift all has shape (B, dim_out, 1, 1)
            scale, shift = time_proj.chunk(2, dim=1) 

            # Modulate features (x_act2 has out_channels, scale/shift have out_channels)
            # FiLM operation: gamma * x + beta
            h = h * (scale + 1) + shift 
        if cond is not None:
            h += cond
        h = self.block2(h)
        return h + self.res_conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 2),
        )

    def forward(self, x):
        return self.conv(x)

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))   
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)  # shape of x: (b, nf, h, w)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x  # shape of out: (b, nf, h, w)

