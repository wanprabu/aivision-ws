from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision.utils import save_image

import torch
import os
import numpy as np

from generator import GeneratorCgan
from discriminator import DiscriminatorCgan
from dataclasses import dataclass
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "cgan.log")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cgan")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# config
@dataclass
class Config:
    n_epochs: int = 200
    batch_size: int = 64
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    n_cpus: int = 12
    n_classes: int = 10
    img_size: int = 32
    latent_dim: int = 100
    channels: int = 1
    sample_interval: int = 400
    img_shape: tuple = (channels, img_size, img_size)

@dataclass
class TrainingState:
    generator: GeneratorCgan = None
    discriminator: DiscriminatorCgan = None
    optimizer_G: torch.optim.Adam = None
    optimizer_D: torch.optim.Adam = None
    adversarial_loss: torch.nn.MSELoss = None
    dataloader: DataLoader = None
    device: torch.device = None
    FloatTensor: torch.Tensor = None
    LongTensor: torch.Tensor = None

@dataclass
class SampleState:
    n_row: int = 5
    batches_done: int = 0
    path: str = OUTPUT_DIR
    generator: GeneratorCgan = None
    device: torch.device = None
    FloatTensor: torch.Tensor = None
    LongTensor: torch.Tensor = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

# Check for CUDA
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# Loss function
adversarial_loss = torch.nn.MSELoss()

def sample_image(param: SampleState, opt: Config = Config()):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(param.FloatTensor(np.random.normal(0, 1, (param.n_row ** 2, opt.latent_dim))))

    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(param.n_row) for num in range(param.n_row)])
    labels = Variable(param.LongTensor(labels))
    gen_imgs = param.generator(z, labels)

    save_image(gen_imgs.data, "%s/%d.png" % (param.path, param.batches_done), nrow=param.n_row, normalize=True)

# ----------
#  Training
# ----------

def train(param: TrainingState, opt: Config = Config()):
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(param.dataloader):

            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(torch.ones(batch_size, 1, device=param.device, dtype=torch.float32), requires_grad=False)
            fake = Variable(torch.zeros(batch_size, 1, device=param.device, dtype=torch.float32), requires_grad=False)

            # old code
            # valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
            # fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.to(param.device).float())
            labels = Variable(labels.to(param.device).long())

            # old code
            # real_imgs = Variable(imgs.type(FloatTensor))
            # labels = Variable(labels.type(LongTensor))

            # -----------------
            #  Train Generator
            # -----------------

            param.optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(param.FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            gen_labels = Variable(param.LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

            # Generate a batch of images
            gen_imgs = param.generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = param.discriminator(gen_imgs, gen_labels)
            g_loss = param.adversarial_loss(validity, valid)

            g_loss.backward()
            param.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            param.optimizer_D.zero_grad()

            # Loss for real images
            validity_real = param.discriminator(real_imgs, labels)
            d_real_loss = param.adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = param.discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = param.adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            param.optimizer_D.step()

            if (epoch % 5 == 0) & (i % 100 == 0):
                logging.info(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(param.dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(param.dataloader) + i

            sample_state = SampleState(
                n_row=10,
                batches_done=batches_done,
                generator=param.generator,
                device=param.device,
                FloatTensor=param.FloatTensor,
                LongTensor=param.LongTensor,
            )

            if batches_done % opt.sample_interval == 0:
                sample_image(sample_state)


def main():
    logging.info(f"===== Starting CGAN Training =====")

    # Initialize generator and discriminator
    generator = GeneratorCgan(Config.latent_dim, Config.img_shape, Config.n_classes)
    discriminator = DiscriminatorCgan(Config.img_shape, Config.n_classes)

    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    if cuda:
        logging.info("Training on GPU")
    else:
        logging.info("Training on CPU")

    # Configure data loader
    os.makedirs("../dataset/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../dataset/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(Config.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=Config.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Config.lr, betas=(Config.b1, Config.b2))

    FloatTensor = lambda data: torch.tensor(data, dtype=torch.float32, device=device)
    LongTensor = lambda data: torch.tensor(data, dtype=torch.long, device=device)

    # Create a TrainingState instance
    training_state = TrainingState(
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        adversarial_loss=adversarial_loss,
        dataloader=dataloader,
        device=device,
        FloatTensor=FloatTensor,
        LongTensor=LongTensor,
    )

    train(training_state)


if __name__ == "__main__":
    main()