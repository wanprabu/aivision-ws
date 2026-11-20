from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision.utils import save_image

import torch.autograd as autograd
import torch
import os
import numpy as np

from generator import GeneratorWgan_gp
from discriminator import DiscriminatorWgan_gp
from dataclasses import dataclass
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "wgan_gp.log")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "wgan_gp")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# config
@dataclass
class Config:
    n_epochs: int = 40
    batch_size: int = 64
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    n_cpus: int = 12
    latent_dim: int = 100
    img_size: int = 32
    channels: int = 1
    n_critic: int = 5
    clip_value: float = 0.01
    sample_interval: int = 400
    img_shape: tuple = (channels, img_size, img_size)

@dataclass
class TrainingState:
    generator: GeneratorWgan_gp = None
    discriminator: DiscriminatorWgan_gp = None
    optimizer_G: torch.optim.Adam = None
    optimizer_D: torch.optim.Adam = None
    dataloader: DataLoader = None
    Tensor: torch.Tensor = None

@dataclass
class GradientState:
    Tensor: torch.Tensor = None
    discriminator: DiscriminatorWgan_gp = None
    real_samples: torch.Tensor = None
    fake_samples: torch.Tensor = None


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

# Check for CUDA
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda else "cpu")

# Loss function
# adversarial_loss = torch.nn.MSELoss()

def tensor_factory(*args, **kwargs):
    if len(args) >= 1 and all(isinstance(arg, (int, np.integer)) for arg in args):
        return torch.empty(*args, device=device, dtype=torch.float32)
    return torch.tensor(*args, device=device, dtype=torch.float32, **kwargs)


def compute_gradient_penalty(param: GradientState):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = param.Tensor(np.random.random((param.real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * param.real_samples + ((1 - alpha) * param.fake_samples)).requires_grad_(True)
    d_interpolates = param.discriminator(interpolates)
    fake = Variable(param.Tensor(param.real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

# ----------
#  Training
# ----------

def train(param: TrainingState, opt: Config = Config()):
    batches_done = 0

    # Loss weight for gradient penalty
    lambda_gp = 10

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(param.dataloader):

            # Configure input
            real_imgs = Variable(imgs.to(device, dtype=torch.float32))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            param.optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(tensor_factory(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = param.generator(z)

            # Real images
            real_validity = param.discriminator(real_imgs)
            # Fake images
            fake_validity = param.discriminator(fake_imgs)
            # Gradient penalty
            gradient_state = GradientState(
                Tensor=param.Tensor,
                discriminator=param.discriminator,
                real_samples=real_imgs.data,
                fake_samples=fake_imgs.data,
            )
            gradient_penalty = compute_gradient_penalty(gradient_state)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            param.optimizer_D.step()

            param.optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = param.generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = param.discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                param.optimizer_G.step()

                if (epoch % 5 == 0) & (i % 100 == 0):
                    logging.info(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, opt.n_epochs, i, len(param.dataloader), d_loss.item(), g_loss.item())
                    )

                if batches_done % opt.sample_interval == 0:
                    # save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                    save_image(fake_imgs.data[:25], "%s/%d.png" % (OUTPUT_DIR, batches_done), nrow=5, normalize=True)

                batches_done += opt.n_critic


def main():
    logging.info(f"===== Starting WGAN-GP Training =====")

    if cuda:
        logging.info("Training on GPU")
        torch.cuda.init()
        torch.cuda.set_device(0)
    else:
        logging.info("Training on CPU")

    # Initialize generator and discriminator
    generator = GeneratorWgan_gp(Config.latent_dim, Config.img_shape)
    discriminator = DiscriminatorWgan_gp(Config.img_shape)

    generator.to(device)
    discriminator.to(device)

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

    Tensor = tensor_factory

    # Create a TrainingState instance
    training_state = TrainingState(
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        dataloader=dataloader,
        Tensor=Tensor,
    )

    train(training_state)


if __name__ == "__main__":
    main()