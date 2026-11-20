import os

import torch
import torchvision
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "check.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
)

logging.info(f"CUDA Available: {torch.cuda.is_available()}")
logging.info(f"Torchvision Version: {torchvision.__version__}")
