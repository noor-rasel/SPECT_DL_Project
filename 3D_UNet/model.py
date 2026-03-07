import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Consists of two convolutional layers with instance norm 3d normalization 
    followed by a ReLU activation function.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):  
        return self.net(x)  


class encoder(nn.Module):
    """ Encoder (downsampling) part of the UNet architecture. """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


def pad_to_match(x, ref):
    dd = ref.size(2) - x.size(2)
    dh = ref.size(3) - x.size(3)
    dw = ref.size(4) - x.size(4)

    return F.pad(
        x,
        (dw//2, dw-dw//2,
         dh//2, dh-dh//2,
         dd//2, dd-dd//2)
    )


class decoder(nn.Module):
    """ The decoder (upsampling) part of the UNet architecture. """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = pad_to_match(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """ UNet architecture. """
    def __init__(self, in_ch=2, out_ch=1, base=32):
        super().__init__()
        self.in_conv = DoubleConv(in_ch, base)

        self.down1 = encoder(base, base*2)
        self.down2 = encoder(base*2, base*4)
        self.down3 = encoder(base*4, base*8)
        self.down4 = encoder(base*8, base*16)

        self.up1 = decoder(base*16, base*8)
        self.up2 = decoder(base*8, base*4)
        self.up3 = decoder(base*4, base*2)
        self.up4 = decoder(base*2, base)

        self.out_conv = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        s1 = self.in_conv(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        
        bottleneck = self.down4(s4)

        x = self.up1(bottleneck, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)

        return self.out_conv(x)