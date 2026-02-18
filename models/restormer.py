import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_conv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1, groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gating mechanism
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        # Norm 1 + Attention
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)
        # Norm 2 + FFN
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        return x

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.body = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False) # PixelShuffle or Deconv

    def forward(self, x):
        # Using functional interpolate + conv is often smoother than TransposedConv
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) 
        # Note: Actual Restormer uses PixelShuffle, but interpolate is fine for starters. 
        # If you want exact official structure, we can add PixelShuffle later.

class Restormer(nn.Module):
    def __init__(self, 
                 inp_channels=3, 
                 out_channels=3, 
                 dim=48, 
                 num_blocks=[4, 6, 6, 8], 
                 num_refinement_blocks=4, 
                 heads=[1, 2, 4, 8], 
                 ffn_expansion_factor=2.66):
        super(Restormer, self).__init__()

        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        self.down_layers = nn.ModuleList([]) # Downsampling

        # Encoder
        for i in range(4):
            self.encoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim * (2**i), heads[i], ffn_expansion_factor) for _ in range(num_blocks[i])
            ]))
            if i < 3:
                self.down_layers.append(DownSample(dim * (2**i), dim * (2**(i+1))))

        # Bottleneck - handled by the last encoder layer usually, or we can add explicit bottleneck
        
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([]) # Upsampling
        self.skip_convs = nn.ModuleList([])

        for i in range(3): # 3 upsampling stages
            # Input dim is next level, Output is current level
            self.up_layers.append(nn.Sequential(
                nn.Conv2d(dim * (2**(3-i)), dim * (2**(3-i)) * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            self.skip_convs.append(nn.Conv2d(dim * (2**(2-i)) * 2, dim * (2**(2-i)), kernel_size=1, bias=False))
            
            self.decoder_layers.append(nn.Sequential(*[
                TransformerBlock(dim * (2**(2-i)), heads[2-i], ffn_expansion_factor) for _ in range(num_blocks[2-i])
            ]))

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor) for _ in range(num_refinement_blocks)
        ])

        self.output_layer = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        inp_img = x
        x = self.patch_embed(x)
        
        encoder_outs = []
        
        # Encoder Level 1
        x = self.encoder_layers[0](x)
        encoder_outs.append(x)
        x = self.down_layers[0](x)
        
        # Encoder Level 2
        x = self.encoder_layers[1](x)
        encoder_outs.append(x)
        x = self.down_layers[1](x)

        # Encoder Level 3
        x = self.encoder_layers[2](x)
        encoder_outs.append(x)
        x = self.down_layers[2](x)

        # Encoder Level 4 (Bottleneck)
        x = self.encoder_layers[3](x)

        # Decoder Level 3
        x = self.up_layers[0](x)
        x = torch.cat([x, encoder_outs[2]], dim=1)
        x = self.skip_convs[0](x)
        x = self.decoder_layers[0](x)

        # Decoder Level 2
        x = self.up_layers[1](x)
        x = torch.cat([x, encoder_outs[1]], dim=1)
        x = self.skip_convs[1](x)
        x = self.decoder_layers[1](x)

        # Decoder Level 1
        x = self.up_layers[2](x)
        x = torch.cat([x, encoder_outs[0]], dim=1)
        x = self.skip_convs[2](x)
        x = self.decoder_layers[2](x)

        # Refinement
        x = self.refinement(x)
        x = self.output_layer(x)

        return x + inp_img # Residual learning (Input + Restoration)