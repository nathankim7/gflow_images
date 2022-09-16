import numpy as np
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
from models import ImageGFN
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DataLoader, Subset
import wandb

# CHANGE API KEY
wandb.login(key="e9d0f0abd4a0b92aa26694bdecd67aa7d57b76d6")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = MNIST(root="../datasets/MNIST", train=True, download=True, transform=ToTensor())
train_data = Subset(data, list(np.random.choice(60000, 5000, replace=False)))
train_loader = DataLoader(train_data, batch_size=32, num_workers=4)

model = ImageGFN(
    n_channels=1, img_dim=28, output_dim=48, num_gaussians=96, lr=8 * 1e-3, wavelet=True
)
wandb_logger = WandbLogger(project="gflow_images", log_model=True)
lr_monitor = LearningRateMonitor(logging_interval="epoch")
trainer = pl.Trainer(
    # overfit_batches=10,
    max_epochs=5,
    logger=wandb_logger,
    accelerator="gpu",
    auto_lr_find="lr",
    gradient_clip_val=0.9,
    gradient_clip_algorithm="norm",
    accumulate_grad_batches=2,
    callbacks=[lr_monitor],
)
trainer.fit(model=model, train_dataloaders=train_loader)
x = model().cpu().numpy()
plt.imshow(x[0])
plt.savefig("./gen.png")
#
# w = pywt.Wavelet('haar')
# x = [9, 7, 3, 5, 6, 10, 2, 6, 4, 1]
# y = pywt.wavedec(x, w)
# print(y)
