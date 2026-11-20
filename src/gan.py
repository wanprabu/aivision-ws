from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision.utils import save_image

import torch
import os
import numpy as np

from bolt_dataset import BoltDataset
from generator import Generator
from discriminator import Discriminator
from dataclasses import dataclass
import logging

# config
@dataclass
class Config:
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    n_cpus: int = 12
    img_size: int = 28 # 28 for MNIST, 32 for CIFAR-10, 64 for CelebA, 256 for custom datasets
    latent_dim: int = 100
    channels: int = 1 # 1 gray-scale, 3 RGB
    sample_interval: int = 400
    img_shape: tuple = (channels, img_size, img_size)

@dataclass
class TrainingState:
    generator: Generator = None
    discriminator: Discriminator = None
    optimizer_G: torch.optim.Adam = None
    optimizer_D: torch.optim.Adam = None
    adversarial_loss: torch.nn.BCELoss = None
    dataloader: DataLoader = None
    Tensor: torch.Tensor = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "mnist")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "gan")
LOG_FILE = os.path.join(LOG_DIR, "gan.log")

# Create sample and checkpoint directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

# Check for CUDA
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul_allow_tf32 = True

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# ----------
#  Training
# ----------

def train(param: TrainingState, opt: Config = Config()):
    scaler_G = torch.amp.GradScaler('cuda', enabled=cuda)
    scaler_D = torch.amp.GradScaler('cuda', enabled=cuda)

    # Adversarial ground truths
    valid = torch.ones(opt.batch_size, 1, device=device, requires_grad=False)
    fake = torch.zeros(opt.batch_size, 1, device=device, requires_grad=False)

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(param.dataloader):

            # valid = Variable(param.Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            # fake = Variable(param.Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = imgs.to(device, non_blocking=True)
            # real_imgs = Variable(imgs.type(param.Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            param.optimizer_G.zero_grad(set_to_none=True)

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
            # z = Variable(param.Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                gen_imgs = param.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = param.adversarial_loss(param.discriminator(gen_imgs), valid)

            # scaler_G.scale(g_loss).backward()
            # scaler_G.step(param.optimizer_G)
            # scaler_G.update()

            g_loss.backward()
            param.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            param.optimizer_D.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # Measure discriminator's ability to classify real from generated samples
                real_loss = param.adversarial_loss(param.discriminator(real_imgs), valid)
                fake_loss = param.adversarial_loss(param.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

            # scaler_D.scale(d_loss).backward()
            # scaler_D.step(param.optimizer_D)
            # scaler_D.update()

            d_loss.backward()
            param.optimizer_D.step()

            if (epoch % 5 == 0) & (i % 100 == 0):
                logging.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(param.dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(param.dataloader) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "%s/%d.png" % (OUTPUT_DIR, batches_done), nrow=5, normalize=True)


def main():
    logging.info(f"===== Starting GAN Training =====")

    # Initialize generator and discriminator
    generator = Generator(Config.latent_dim, Config.img_shape)
    discriminator = Discriminator(Config.img_shape)

    if cuda:
        generator.to(device)
        discriminator.to(device)
        adversarial_loss.to(device)

        # generator = torch.compile(generator, mode="max-autotune", fullgraph=True)
        # discriminator = torch.compile(discriminator, mode="max-autotune", fullgraph=True)

        logging.info("Training on GPU")

    # Configure data loader

    os.makedirs(DATASET_DIR, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            DATASET_DIR,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(Config.img_size), 
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        [0.5], [0.5]
                    ) 
                ]
            ),
        ),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.n_cpus,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # Create a TrainingState instance
    training_state = TrainingState(
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        adversarial_loss=adversarial_loss,
        dataloader=dataloader,
        Tensor=Tensor
    )

    train(training_state)

    logging.info(f"===== Finished GAN Training =====")


if __name__ == "__main__":
    main()