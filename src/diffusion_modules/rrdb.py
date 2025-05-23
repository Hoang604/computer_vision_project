import functools
import torch
from torch import nn
import torch.nn.functional as F
from src.utils.network_comopents import RRDB

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

