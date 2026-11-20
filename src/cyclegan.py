import argparse
import os
import numpy as np
# import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

from dataclasses import dataclass
import logging

# config
@dataclass
class Config:
    epoch: int = 0
    n_epochs: int = 20
    dataset_name: str = "apple2orange"
    batch_size: int = 1
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    decay_epoch: int = 10
    n_cpu: int = 12
    img_height: int = 256
    img_width: int = 256
    channels: int = 3
    sample_interval: int = 100
    checkpoint_interval: int = -1
    n_residual_blocks: int = 9
    lambda_cyc: float = 10.0
    lambda_id: float = 5.0
    input_shape: tuple = (channels, img_height, img_width)

@dataclass
class TrainingState:
    G_AB: GeneratorResNet = None
    G_BA: GeneratorResNet = None
    D_A: Discriminator = None
    D_B: Discriminator = None
    optimizer_G: torch.optim.Adam = None
    optimizer_D_A: torch.optim.Adam = None
    optimizer_D_B: torch.optim.Adam = None
    lr_scheduler_G: torch.optim.lr_scheduler.LambdaLR = None
    lr_scheduler_D_A: torch.optim.lr_scheduler.LambdaLR = None
    lr_scheduler_D_B: torch.optim.lr_scheduler.LambdaLR = None
    fake_A_buffer: ReplayBuffer = None
    fake_B_buffer: ReplayBuffer = None
    Tensor: torch.Tensor = None

@dataclass
class SampleState:
    batches_done: int = 0
    G_AB: GeneratorResNet = None
    G_BA: GeneratorResNet = None
    Tensor: torch.Tensor = None


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", Config.dataset_name)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cyclegan")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "saved_models", Config.dataset_name)
LOG_FILE = os.path.join(LOG_DIR, "cyclegan.log")

# Create sample and checkpoint directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Image transformations
transforms_ = [
    transforms.Resize(int(Config.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((Config.img_height, Config.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset(DATASET_DIR, transforms_=transforms_, unaligned=True),
    batch_size=Config.batch_size,
    shuffle=True,
    num_workers=Config.n_cpu,
)

# Test data loader
val_dataloader = DataLoader(
    ImageDataset(DATASET_DIR, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)

# os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
# os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

def sample_images(param: SampleState, opt: Config = Config()):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    param.G_AB.eval()
    param.G_BA.eval()

    real_A = Variable(imgs["A"].type(param.Tensor))
    fake_B = param.G_AB(real_A)
    real_B = Variable(imgs["B"].type(param.Tensor))
    fake_A = param.G_BA(real_B)

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)    
    save_image(image_grid, "%s/%d.png" % (OUTPUT_DIR, param.batches_done), normalize=False)


# ----------
#  Training
# ----------

def train(param: TrainingState, opt: Config = Config()):
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):

            # Set model input
            real_A = Variable(batch["A"].type(param.Tensor))
            real_B = Variable(batch["B"].type(param.Tensor))

            # Adversarial ground truths
            valid = Variable(
                torch.tensor(
                    np.ones((real_A.size(0), *param.D_A.output_shape)),
                    dtype=torch.float32,
                    device=real_A.device,
                ),
                requires_grad=False,
            )
            fake = Variable(
                torch.tensor(
                    np.zeros((real_A.size(0), *param.D_A.output_shape)),
                    dtype=torch.float32,
                    device=real_A.device,
                ),
                requires_grad=False,
            )

            # ------------------
            #  Train Generators
            # ------------------

            param.G_AB.train()
            param.G_BA.train()

            param.optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(param.G_BA(real_A), real_A)
            loss_id_B = criterion_identity(param.G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = param.G_AB(real_A)
            loss_GAN_AB = criterion_GAN(param.D_B(fake_B), valid)
            fake_A = param.G_BA(real_B)
            loss_GAN_BA = criterion_GAN(param.D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = param.G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = param.G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            param.optimizer_G.step()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            param.optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(param.D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = param.fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(param.D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            param.optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            param.optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(param.D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = param.fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(param.D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            param.optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            if (epoch % 5 == 0) & (i % 100 == 0):
                logging.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                    % 
                    (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                    )
                )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_state = SampleState(
                    batches_done=batches_done,
                    G_AB=param.G_AB,
                    G_BA=param.G_BA,
                    Tensor=param.Tensor,
                )
                sample_images(sample_state)

        # Update learning rates
        param.lr_scheduler_G.step()
        param.lr_scheduler_D_A.step()
        param.lr_scheduler_D_B.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(param.G_AB.state_dict(), "%s/G_AB_%d.pth" % (MODEL_DIR, epoch))
            torch.save(param.G_BA.state_dict(), "%s/G_BA_%d.pth" % (MODEL_DIR, epoch))
            torch.save(param.D_A.state_dict(), "%s/D_A_%d.pth" % (MODEL_DIR, epoch))
            torch.save(param.D_B.state_dict(), "%s/D_B_%d.pth" % (MODEL_DIR, epoch))


def main():
    logging.info(f"===== Starting CycleGAN Training =====")

    cuda = torch.cuda.is_available()

    # Initialize generator and discriminator
    G_AB = GeneratorResNet(Config.input_shape, Config.n_residual_blocks)
    G_BA = GeneratorResNet(Config.input_shape, Config.n_residual_blocks)
    D_A = Discriminator(Config.input_shape)
    D_B = Discriminator(Config.input_shape)

    if cuda:
        logging.info("Training on GPU")

        G_AB = G_AB.cuda()
        G_BA = G_BA.cuda()
        D_A = D_A.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    if Config.epoch != 0:
        # Load pretrained models
        G_AB.load_state_dict(torch.load("%s/G_AB_%d.pth" % (MODEL_DIR, Config.epoch)))
        G_BA.load_state_dict(torch.load("%s/G_BA_%d.pth" % (MODEL_DIR, Config.epoch)))
        D_A.load_state_dict(torch.load("%s/D_A_%d.pth" % (MODEL_DIR, Config.epoch)))
        D_B.load_state_dict(torch.load("%s/D_B_%d.pth" % (MODEL_DIR, Config.epoch)))
    else:
        # Initialize weights
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=Config.lr, betas=(Config.b1, Config.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(Config.n_epochs, Config.epoch, Config.decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(Config.n_epochs, Config.epoch, Config.decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(Config.n_epochs, Config.epoch, Config.decay_epoch).step
    )

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    training_state = TrainingState(
        G_AB=G_AB,
        G_BA=G_BA,
        D_A=D_A,
        D_B=D_B,
        optimizer_G=optimizer_G,
        optimizer_D_A=optimizer_D_A,
        optimizer_D_B=optimizer_D_B,
        lr_scheduler_G=lr_scheduler_G,
        lr_scheduler_D_A=lr_scheduler_D_A,
        lr_scheduler_D_B=lr_scheduler_D_B,
        fake_A_buffer=fake_A_buffer,
        fake_B_buffer=fake_B_buffer,
        Tensor=Tensor,
    )

    train(training_state)

if __name__ == "__main__":
    main()