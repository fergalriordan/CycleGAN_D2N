import torch 
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
import os
import argparse
import random
import os
import numpy as np

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import preprocess_data as ppd

from models import generator as gen
from models import encoder as enc
from models import sharing_generator as sh_gen
from models import unet as un
from models import unet_encoder as un_enc
from models import unet_decoder as un_dec
from models import unet_resnet18_encoder as un_res

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

DIR = "../data/train"
NIGHT_GT_STATS_PATH = '../data/inception_stats/night/night_statistics.npz'
DAY_GT_STATS_PATH = '../data/inception_stats/day/day_statistics.npz'
FAKE_N_PATH = '../data/inception_images/night'
FAKE_D_PATH = '../data/inception_images/day'
SIZE = 299 # generate images with the same dimensions as the training data for the inception model

# Mean and standard deviation of day and night images
day_means = [0.5, 0.5, 0.5]
night_means = [0.5, 0.5, 0.5]
day_stds = [0.5, 0.5, 0.5]
night_stds = [0.5, 0.5, 0.5]

transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(), 
    ],
    additional_targets={"image0": "image"},
)

def load_checkpoint_for_testing(checkpoint_file, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval() # set model to evaluation mode
    return model

def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["simple", "sharing", "unet", "sharing_unet", "pretrained_encoder"], help="Type of generator to use: simple, sharing, unet, sharing_unet or pretrained_encoder"
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the checkpoints"
    )
    parser.add_argument(
        "--epoch", type=int, help="Epoch number of the checkpoint to load"
    )
    return parser.parse_args()

def main():

    args = parse_arguments()

    model_type = args.model_type
    if model_type not in ["simple", "sharing", "unet", "sharing_unet", "pretrained_encoder"]:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if model_type == "simple":
        gen_N = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
        gen_D = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)

    elif model_type == 'sharing':
        encoder = enc.Encoder(img_channels=3, num_features=64).to(DEVICE)
        gen_N = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)
        gen_D = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)

    elif model_type == 'unet':
        gen_N = un.UNet(input_channel=3, output_channel=3).to(DEVICE)
        gen_D = un.UNet(input_channel=3, output_channel=3).to(DEVICE)

    elif model_type == 'sharing_unet':
        encoder = un_enc.UNet_Encoder(input_channel=3).to(DEVICE)
        gen_N = un_dec.UNet_Decoder(encoder, output_channel=3).to(DEVICE)
        gen_D = un_dec.UNet_Decoder(encoder, output_channel=3).to(DEVICE)

    elif model_type == 'pretrained_encoder':
        gen_N = un_res.UnetResNet18(output_channels=3).to(DEVICE)
        gen_D = un_res.UnetResNet18(output_channels=3).to(DEVICE)

    model_path = args.model_path
    epoch = args.epoch

    gen_N_checkpoint = model_path + f"gen_n_{epoch}.pth.tar"
    gen_D_checkpoint = model_path + f"gen_d_{epoch}.pth.tar"

    load_checkpoint_for_testing(gen_N_checkpoint, gen_N)
    load_checkpoint_for_testing(gen_D_checkpoint, gen_D)   

    day_transforms = ppd.set_val_transforms(day_means, day_stds)
    night_transforms = ppd.set_val_transforms(night_means, night_stds)

    dataset = ppd.DayNightDataset(
        root_day=DIR + "/day",
        root_night=DIR + "/night",
        size=SIZE,
        #transform=transforms,
        day_transform=day_transforms,
        night_transform=night_transforms,
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
    )
    
    day_images = []
    night_images = []
    generated_night_images = []
    generated_day_images = []

    for idx, (night_img, day_img) in enumerate(loader):
        day_img = day_img.to(DEVICE)
        night_img = night_img.to(DEVICE)

        with torch.no_grad():
            fake_night = gen_N(day_img)
            fake_day = gen_D(night_img)
            
        day_images.append(day_img)
        night_images.append(night_img)
        generated_night_images.append(fake_night)
        generated_day_images.append(fake_day)
        
    _ = torch.manual_seed(42)

    kid_D2N = KernelInceptionDistance(subsets=5, subset_size=50, normalize=True).to(DEVICE)
    kid_N2D = KernelInceptionDistance(subsets=5, subset_size=50, normalize=True).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        kid_D2N.update(night, real=True)
        kid_D2N.update(gen_night, real=False)
        
    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        kid_N2D.update(day, real=True)
        kid_N2D.update(gen_day, real=False)

    kid_D2N_mean, kid_D2N_std = kid_D2N.compute()
    kid_N2D_mean, kid_N2D_std = kid_N2D.compute()
    
    print(f"Kernel Inception Distance Mean (D2N): {kid_D2N_mean.item()}, Standard Deviation: {kid_D2N_std.item()}")
    print(f"Kernel Inception Distance Mean (N2D): {kid_N2D_mean.item()}, Standard Deviation: {kid_N2D_std.item()}")

    fid_D2N = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid_N2D = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        fid_D2N.update(night, real=True)
        fid_D2N.update(gen_night, real=False)

    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        fid_N2D.update(day, real=True)
        fid_N2D.update(gen_day, real=False)

    fid_D2N_mean = fid_D2N.compute()
    fid_N2D_mean = fid_N2D.compute()

    print(f"Frechet Inception Distance Mean (D2N): {fid_D2N_mean.item()}")
    print(f"Frechet Inception Distance Mean (N2D): {fid_N2D_mean.item()}")

if __name__ == "__main__":
    seed_everything()
    main()