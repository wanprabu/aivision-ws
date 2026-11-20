import argparse
import os
import numpy as np
# import math
import itertools
import datetime
import time

from bolt_dataset import BoltDataset
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from generator import GeneratorDcgan
from discriminator import DiscriminatorDcgan

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataclasses import dataclass
import logging

# config
@dataclass
class Config:
    n_epochs: int = 300
    batch_size: int = 32
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    decay_epoch: int = 10
    n_cpu: int = 12
    latent_dim: int = 100
    img_size: int = 256
    channels: int = 3
    sample_interval: int = 400
    img_shape: tuple = (channels, img_size, img_size)

@dataclass
class TrainingState:
    generator: GeneratorDcgan = None
    discriminator: DiscriminatorDcgan = None
    optimizer_G: torch.optim.Adam = None
    optimizer_D: torch.optim.Adam = None
    dataloader: DataLoader = None
    Tensor: torch.Tensor = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "bolts", "threebolts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "dcgan2")
LOG_FILE = os.path.join(LOG_DIR, "dcgan.log")

# Create sample and checkpoint directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

# Loss function
# adversarial_loss = torch.nn.BCELoss()
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul_allow_tf32 = True

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# ----------
#  Training
# ----------

def train(param: TrainingState, opt: Config = Config()):
    device = next(param.generator.parameters()).device

    scaler_G = torch.amp.GradScaler(device=device, enabled=True)
    scaler_D = torch.amp.GradScaler(device=device, enabled=True)

    # Adversarial ground truths
    valid = torch.ones((opt.batch_size, 1), device=device, requires_grad=False)
    fake = torch.zeros((opt.batch_size, 1), device=device, requires_grad=False)

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(param.dataloader):

            # Configure input
            real_imgs = imgs.to(device, non_blocking=True)

            # -----------------
            #  Train Generator
            # -----------------

            param.optimizer_G.zero_grad(set_to_none=True)

            # Sample noise as generator input
            z = torch.randn((imgs.shape[0], opt.latent_dim), device=device)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Generate a batch of images
                gen_imgs = param.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(param.discriminator(gen_imgs), valid)
            
            # scaler_G.scale(g_loss).backward()
            # scaler_G.step(param.optimizer_G)
            # scaler_G.update()

            g_loss.backward()
            param.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            param.optimizer_D.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(param.discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(param.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

            # scaler_D.scale(d_loss).backward()
            # scaler_D.step(param.optimizer_D)
            # scaler_D.update()

            d_loss.backward()
            param.optimizer_D.step()

            # Print log
            if (epoch % 5 == 0) & (i % 100 == 0):
                logging.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(param.dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(param.dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%s/%d.png" % (OUTPUT_DIR, batches_done), nrow=5, normalize=True)


def main():
    logging.info(f"===== Starting DCGAN Training =====")

    # Configure data loader
    dataset = BoltDataset(
        root_dir=DATASET_DIR,
        transform=transforms.Compose(
            [
                # transforms.Resize(Config.img_size), 
                transforms.ToTensor(), 
                transforms.Normalize(
                    [0.5, 0.5, 0.5], 
                    [0.5, 0.5, 0.5]
                ) 
            ]
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.n_cpu,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    cuda = torch.cuda.is_available()
    device = torch.device("cpu")

    # Initialize generator and discriminator
    generator = GeneratorDcgan(Config.latent_dim, Config.img_shape)
    discriminator = DiscriminatorDcgan(Config.img_shape)

    if cuda:
        logging.info("Training on GPU")
        device = torch.device("cuda")
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)

        # generator = torch.compile(generator, mode="max-autotune", fullgraph=True)
        # discriminator = torch.compile(discriminator, mode="max-autotune", fullgraph=True)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # logging.info(f"Dataset size: {len(dataset)} images")
    # logging.info(f"Dataset shape: {dataset[0].shape}")

    # print(np.prod(Config.img_shape), " < size of image shape")

    # z = Variable(torch.zeros(Config.batch_size, Config.latent_dim, device=device, dtype=torch.float32), requires_grad=False)
    # fake_img = generator(z)
    # print(fake_img.shape, " < shape of generated image")

    # gen_params = sum(p.numel() for p in generator.parameters() )
    # disc_params = sum(p.numel() for p in discriminator.parameters() )

    # print(f"Generator parameters: {gen_params}")
    # print(f"Discriminator parameters: {disc_params}")

    # print(fake_img.min(), fake_img.max(), " < min and max of generated image")

    # print(discriminator, " < discriminator architecture")

    # real = next(iter(dataloader))
    # print(discriminator(real).shape, " < discriminator output shape for real images")

    # exit(0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    training_state = TrainingState(
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        dataloader=dataloader,
        Tensor=Tensor,
    )

    train(training_state)

    logging.info(f"===== Finished DCGAN Training =====")

if __name__ == "__main__":
    main()