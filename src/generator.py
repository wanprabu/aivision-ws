import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

latent_dim = None
img_shape = None
n_classes = None
init_size = None
conv = None

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class GeneratorCgan(nn.Module):
    def __init__(self, latent_dim, img_shape, n_classes):

        super(GeneratorCgan, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_classes = n_classes

        self.label_emb = nn.Embedding(self.n_classes, self.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim + self.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class GeneratorWgan_gp(nn.Module):
    def __init__(self, latent_dim, img_shape, conv=False):
        super(GeneratorWgan_gp, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.conv = conv

        if conv:
            self.init_size = 8 # img_shape[1] // 4  # Initial size before upsampling

            self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))
            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(128),
    
                # Upsampling
                # 8 -> 16
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                # 16 -> 32
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
    
                # 32 -> 64
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
    
                # 64 -> 128
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
    
                # 128 -> 256
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
    
                nn.Conv2d(32, self.img_shape[0], 3, stride=1, padding=1),
                nn.Tanh(),
            )

        else:
            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))

                return layers

            self.model = nn.Sequential(
                *block(self.latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(self.img_shape))),
                nn.Tanh()
            )

    def forward(self, z):
        if not self.conv:
            img = self.model(z)
            img = img.view(img.size(0), *self.img_shape)
        else:
            out = self.l1(z)
            out = out.view(out.shape[0], 128, self.init_size, self.init_size)
            img = self.conv_blocks(out)

        return img

class GeneratorDcgan(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(GeneratorDcgan, self).__init__()

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.init_size = 8 # img_shape[1] // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            # Upsampling
            # 8 -> 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16 -> 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 -> 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 -> 128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 -> 256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, self.img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img  