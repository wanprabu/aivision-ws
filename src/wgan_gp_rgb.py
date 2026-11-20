from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision.utils import save_image

import contextlib
import torch.autograd as autograd
import torch
import os
import numpy as np

from bolt_dataset import BoltDataset
from generator import GeneratorWgan_gp
from discriminator import DiscriminatorWgan_gp, CriticWganGP
from dataclasses import dataclass
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "bolts", "threebolts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "wgan_gp_2")
LOG_FILE = os.path.join(LOG_DIR, "wgan_gp_rgb.log")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul_allow_tf32 = True

# config
@dataclass
class Config:
    n_epochs: int = 300
    batch_size: int = 32
    lr: float = 0.0002
    b1: float = 0.5
    b2: float = 0.999
    n_cpu: int = 12
    latent_dim: int = 100
    img_size: int = 256
    channels: int = 3
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
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.init()
    torch.cuda.set_device(0)

device = torch.device("cuda:0" if cuda else "cpu")

# Loss function
# adversarial_loss = torch.nn.MSELoss()

def compute_gradient_penalty(param: GradientState):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.tensor(
        np.random.random((param.real_samples.size(0), 1, 1, 1)),
        dtype=torch.float32,
        device=param.real_samples.device,
    )

    # Get random interpolation between real and fake samples
    interpolates = (alpha * param.real_samples + ((1 - alpha) * param.fake_samples)).requires_grad_(True)
    d_interpolates = param.discriminator(interpolates)

    fake = torch.ones(
        (param.real_samples.shape[0], 1),
        dtype=torch.float32,
        device=param.real_samples.device,
        requires_grad=False,
    )

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
            # real_imgs = Variable(imgs.to(device, dtype=torch.float32))
            real_imgs = imgs.to(device, non_blocking=True)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            param.optimizer_D.zero_grad(set_to_none=True)

            # Sample noise as generator input
            # z = Variable(tensor_factory(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            z = torch.randn((imgs.shape[0], opt.latent_dim), device=device)

            # Generate a batch of images
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                fake_imgs = param.generator(z)

                # Real images
                real_validity = param.discriminator(real_imgs)
                # Fake images
                fake_validity = param.discriminator(fake_imgs)
                # Gradient penalty
                gradient_state = GradientState(
                    Tensor=param.Tensor,
                    discriminator=param.discriminator,
                    real_samples=real_imgs.detach(),
                    fake_samples=fake_imgs.detach(),
                )
                gradient_penalty = compute_gradient_penalty(gradient_state)

                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty


            d_loss.backward()
            param.optimizer_D.step()

            param.optimizer_G.zero_grad(set_to_none=True)

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
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
                    # save_image(fake_imgs.detach()[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                    save_image(fake_imgs.detach()[:25], "%s/%d.png" % (OUTPUT_DIR, batches_done), nrow=5, normalize=True)

                batches_done += opt.n_critic


def main():
    logging.info(f"===== Starting WGAN-GP Training =====")

    # Configure data loader
    dataset = BoltDataset(
        root_dir=DATASET_DIR,
        transform=transforms.Compose(
            [
                # transforms.Resize((250, 250)),
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

    # print(f"Dataset size: {len(dataset)} images")
    # print(f"Dataset shape: {dataset[0].shape}")
    # exit(0)

    # Initialize generator and discriminator
    generator = GeneratorWgan_gp(Config.latent_dim, Config.img_shape, True)
    discriminator = CriticWganGP(Config.img_shape)

    if cuda:
        logging.info("Training on GPU")
        # torch.cuda.init()
        # torch.cuda.set_device(0)
        torch.cuda.current_device()
        device_ctx = torch.cuda.device(device)
    else:
        logging.info("Training on CPU")
        device_ctx = contextlib.nullcontext()

    with device_ctx:
        generator.to(device)
        discriminator.to(device)

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
            dataloader=dataloader,
            Tensor=Tensor,
        )

        train(training_state)

        logging.info(f"===== Finished WGAN-GP Training =====")

    return

    # print(np.prod(Config.img_shape), " < size of image shape")

    # z = Variable(torch.zeros(Config.batch_size, Config.latent_dim, device=device, dtype=torch.float32), requires_grad=False)
    # fake_img = generator(z)
    # print(fake_img.shape, " < shape of generated image")

    # gen_params = sum(p.numel() for p in generator.parameters() )
    # disc_params = sum(p.numel() for p in discriminator.parameters() )

    # print(f"Generator parameters: {gen_params}")
    # print(f"Discriminator parameters: {disc_params}")

    # x = torch.randn(64, 3, 256, 256).to(device)
    # y = discriminator(x)
    # print(y.shape, " < shape of discriminator output for real images")

    # # print(fake_img.min(), fake_img.max(), " < min and max of generated image")

    # print(generator, " < generator architecture")
    # print(discriminator, " < discriminator architecture")

    # # real = next(iter(dataloader))
    # # print(discriminator(real).shape, " < discriminator output shape for real images")

    # exit(0)

if __name__ == "__main__":
    main()