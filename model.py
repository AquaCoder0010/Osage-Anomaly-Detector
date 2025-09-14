import torch
import torch.nn as nn
from torchinfo import summary

def get_bn_layer(bn_type='none', num_features=None, dims=None):
    if bn_type == 'batch':
        return nn.BatchNorm1d(num_features) if dims == 1 else nn.BatchNorm2d(num_features)
    elif bn_type == 'layer':
        return nn.GroupNorm(1, num_features)
    elif bn_type == 'instance':
        return nn.InstanceNorm1d(num_features) if dims == 1 else nn.InstanceNorm2d(num_features, affine=True)
    elif bn_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported normalization layer type: {bn_type}")


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True, upsample_type='bilinear', use_bias=True, bn_type='none', act=nn.ReLU):
        super().__init__()
        self.upsample = upsample
        
        if upsample:
            if upsample_type == 'bilinear':
                self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            else:
                self.upsample_layer = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = get_bn_layer(bn_type, out_channels)
        self.act1 = act()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = get_bn_layer(bn_type, out_channels)
        self.act2 = act()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=use_bias)
        self.bn_skip = get_bn_layer(bn_type, out_channels)
        
        self.bn_final = get_bn_layer(bn_type, out_channels)
        self.act_final = act()

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
            
        skip = self.bn_skip(self.conv_skip(x))
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.act2(self.bn2(self.conv2(h)))
        h = self.conv3(h)
        
        return self.act_final(self.bn_final(h + skip))

#DBlock(3, 1 * channels, use_bias=use_bias, bn_type=bn_type, act=act),        # -> 64x64 if image_size=128
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, use_bias=True, bn_type='none', act=nn.ReLU):
        super().__init__()
        self.pool = pool
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn1 = get_bn_layer(bn_type, out_channels)
        self.act1 = act()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.bn2 = get_bn_layer(bn_type, out_channels)
        self.act2 = act()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=use_bias)
        self.bn_skip = get_bn_layer(bn_type, out_channels)

        self.bn_final = get_bn_layer(bn_type, out_channels)
        self.act_final = act()
        
        if self.pool:
            self.pool_layer = nn.AvgPool2d(2)

    def forward(self, x):
        skip = self.bn_skip(self.conv_skip(x))
        
        h = self.act1(self.bn1(self.conv1(x)))
        h = self.act2(self.bn2(self.conv2(h)))
        h = self.conv3(h)
        
        out = self.act_final(self.bn_final(h + skip))
        if self.pool:
            out = self.pool_layer(out)
        return out

class Generator(nn.Module):
    def __init__(self, latent_size, channels=3, upsample_first=True, upsample_type='bilinear', bn_type='none', act_type='lrelu'):
        super().__init__()
        
        use_bias = bn_type == 'none'
        act = lambda: nn.LeakyReLU(0.2, inplace=True) if act_type == 'lrelu' else nn.ReLU(inplace=True)
        
        self.init_dense = nn.Linear(latent_size, 2 * 2 * 16 * channels, bias=use_bias)
        self.init_bn = get_bn_layer(bn_type, num_features=2 * 2 * 16 * channels, dims=1)
        self.init_channels = 16 * channels
        
        self.g_block1 = GBlock(16 * channels, 16 * channels, upsample=upsample_first, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block2 = GBlock(16 * channels, 8 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block3 = GBlock(8 * channels, 4 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block4 = GBlock(4 * channels, 3 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block5 = GBlock(3 * channels, 2 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block6 = GBlock(2 * channels, 1 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)
        self.g_block7 = GBlock(1 * channels, 1 * channels, upsample_type=upsample_type, use_bias=use_bias, bn_type=bn_type, act=act)

        
        self.final_conv = nn.Conv2d(1 * channels, 3, kernel_size=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        print(x.shape);
        x = self.init_dense(x);
        x = self.init_bn(x);        
        # Reshape to (Batch, Channels, Height, Width) for PyTorch conv layers
        x = x.view(-1, self.init_channels, 2, 2)
        x = self.g_block1(x)
        x = self.g_block2(x)
        x = self.g_block3(x)
        x = self.g_block4(x)
        x = self.g_block5(x)
        x = self.g_block6(x)
        x = self.g_block7(x)
        x = self.final_conv(x)

        return self.tanh(x)

class Encoder(nn.Module):
    def __init__(self, image_size, latent_size, channels=3, bn_type='none', act_type='lrelu'):
        super().__init__()
        use_bias = bn_type == 'none'
        act = lambda: nn.LeakyReLU(0.2, inplace=True) if act_type == 'lrelu' else nn.ReLU(inplace=True)

        self.blocks = nn.Sequential(
            DBlock(3, 1 * channels, use_bias=use_bias, bn_type=bn_type, act=act),        # -> 64x64 if image_size=128
            DBlock(1 * channels, 2 * channels, use_bias=use_bias, bn_type=bn_type, act=act), # -> 32x32
            DBlock(2 * channels, 3 * channels, use_bias=use_bias, bn_type=bn_type, act=act), # -> 16x16
            DBlock(3 * channels, 4 * channels, use_bias=use_bias, bn_type=bn_type, act=act), # -> 8x8
            DBlock(4 * channels, 8 * channels, use_bias=use_bias, bn_type=bn_type, act=act), # -> 4x4
            DBlock(8 * channels, 16 * channels, pool=False, use_bias=use_bias, bn_type=bn_type, act=act) # -> 4x4
        )
        
        self.final_dense1 = nn.Linear(16 * channels * 4 * 4, 16 * channels, bias=use_bias)
        self.final_bn = get_bn_layer(bn_type, num_features=16 * channels, dims=1)
        self.final_act = act()
        self.final_dense2 = nn.Linear(16 * channels, latent_size)

    def forward(self, x):
        x = self.blocks(x)
        x = torch.flatten(x, start_dim=1)
        x = self.final_act(self.final_bn(self.final_dense1(x)))
        x = self.final_dense2(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size, latent_size, channels=3, bn_type='none', act_type='lrelu'):
        super().__init__()
        use_bias = bn_type == 'none'
        act = lambda: nn.LeakyReLU(0.2, inplace=True) if act_type == 'lrelu' else nn.ReLU(inplace=True)

        self.latent_path = nn.Sequential(
            nn.Linear(latent_size, 512, bias=use_bias),
            get_bn_layer(bn_type, 512, dims=1),
            act(),
            nn.Linear(512, 512, bias=use_bias),
            get_bn_layer(bn_type, 512, dims=1),
            act(),
            nn.Linear(512, 512, bias=use_bias),
            get_bn_layer(bn_type, 512, dims=1),
            act()
        )
        
        self.image_path = nn.Sequential(
            DBlock(3, 1 * channels, use_bias=use_bias, bn_type=bn_type, act=act),        
            DBlock(1 * channels, 2 * channels, use_bias=use_bias, bn_type=bn_type, act=act),
            DBlock(2 * channels, 3 * channels, use_bias=use_bias, bn_type=bn_type, act=act),
            DBlock(3 * channels, 4 * channels, use_bias=use_bias, bn_type=bn_type, act=act),
            DBlock(4 * channels, 8 * channels, use_bias=use_bias, bn_type=bn_type, act=act),
            DBlock(8 * channels, 16 * channels, pool=False, use_bias=use_bias, bn_type=bn_type, act=act)
        )

        # Common path
        self.common_path = nn.Sequential(
            nn.Linear(16 * channels * 4 * 4 + 512, 16 * channels, bias=use_bias),
            get_bn_layer(bn_type, 16 * channels, dims=1),
            act(),
            nn.Linear(16 * channels, 1)
        )

    def forward(self, image, latent):
        l = self.latent_path(latent)
        x = self.image_path(image)
        x = torch.flatten(x, start_dim=1)
        
        combined = torch.cat([x, l], dim=1)
        return self.common_path(combined)

from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

if __name__ == '__main__':
    LATENT_SIZE = 64
    CHANNELS = 3
    IMG_SIZE = 128
    BATCH_SIZE = 4

    # Instantiate models
    g = Generator(LATENT_SIZE, CHANNELS, upsample_first=False, bn_type='batch')
    e = Encoder(IMG_SIZE, LATENT_SIZE, bn_type='instance')
    d = Discriminator(IMG_SIZE, LATENT_SIZE, bn_type='layer')
    
    # Print summaries
    print("Generator Summary")
    summary(g, input_size=(BATCH_SIZE, LATENT_SIZE))
    
    print("Encoder Summary")
    summary(e, input_size=(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))

    print("Discriminator Summary")
    summary(d, input_size=[(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE), (BATCH_SIZE, LATENT_SIZE)])