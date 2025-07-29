import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    def __init__(self,z_dim=512,w_dim=512,n_layers=8):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(z_dim if i ==0 else w_dim,w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self,z):
        return self.mapping(z)

class AdaIN(nn.Module):
    def __init__(self,channels,w_dim=512):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = nn.Linear(w_dim,channels)
        self.style_bias = nn.Linear(w_dim,channels)

    def forward(self,x,w):
        x = self.instance_norm(x)
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return x * (1+scale) + bias

class StyleBlock(nn.Module):
    def __init__(self,in_channels,out_channels,w_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1)
        self.adaIN = AdaIN(out_channels,w_dim)
        self.noise_weight = nn.Parameter(torch.Tensor(torch.zeros(1)))
        self.bias = nn.Parameter(torch.zeros(1,out_channels,1,1))

    def forward(self,x,w,noise=None):
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        x = self.conv(x)
        if noise is not None:
            noise = torch.randn(x.shape[0],1,x.shape[2],x.shape[3],device=x.device)
        x = x + self.noise_weight * noise
        x = x + self.bias
        x = F.leaky_relu(x,0.2)
        x = self.adaIN(x,w)
        return x

class SytleGANGenerator(nn.Module):
    def __init__(self,z_dim=512,w_dim=512,img_channels=3,max_resolutiuon=64):
        super().__init__()
        self.mapping = MappingNetwork(z_dim=z_dim,w_dim=w_dim)
        self.const_input = nn.Parameter(torch.randn(1,512,4,4))
        self.initial_block = StyleBlock(512,512,w_dim=w_dim)

        self.blocks = nn.ModuleList()
        res = 8
        while res <= max_resolutiuon:
            in_ch = 512 if res //2 <= 16 else int(512/(res//16))
            out_ch = 512 if res //2 <= 16 else int(512/(res//16))
            self.blocks.append(StyleBlock(in_ch,out_ch,w_dim=w_dim))
            res *= 2
        self.to_rgb = nn.Conv2d(3,img_channels,1)
    def forward(self,z,noise=None):
        w = self.mapping(z)
        x = self.const_input.expand(z.shape[0],-1,-1,-1)
        x = self.initial_block(x,w,noise)

        for block in self.blocks:
            x = block(x,w,noise)
        x = self.to_rgb(x)
        return torch.tanh(x)

class SytleGANDiscriminator(nn.Module):
    def __init__(self,img_channels=3,max_resolutiuon=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(img_channels,64,kernel_size=4,stride=2,padding=1),nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1),nn.BatchNorm2d(128),nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1),nn.BatchNorm2d(256),nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten()
        )
        self.classifier = nn.Linear(256*4*4,1)

    def forward(self,x):
        x = self.features(x)
        return self.classifier(x)

# 测试生成
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = SytleGANGenerator().to(device)
z = torch.randn(4, 512).to(device)
with torch.no_grad():
    fake_images = G(z)
print(fake_images.shape)  # [4, 3, 64, 64]