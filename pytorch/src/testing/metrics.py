import torch 
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
import os
import argparse

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import generator as gen
from CycleGAN_D2N.pytorch.src.preprocessing import preprocess_data as ppd
from models import encoder as enc
from models import sharing_generator as sh_gen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

DIR = "../data/train"
SIZE = 299 # generate images with the same dimensions as the training data for the inception model

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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["simple", "sharing"], help="Type of generator to use: simple or sharing"
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
    if model_type not in ["simple", "sharing"]:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if model_type == "simple":
        gen_N = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
        gen_D = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)

    elif model_type == 'sharing':
        encoder = enc.Encoder(img_channels=3, num_features=64).to(DEVICE)
        gen_N = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)
        gen_D = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)

    model_path = args.model_path
    epoch = args.epoch

    gen_N_checkpoint = model_path + f"gen_n_{epoch}.pth.tar"
    gen_D_checkpoint = model_path + f"gen_d_{epoch}.pth.tar"

    load_checkpoint_for_testing(gen_N_checkpoint, gen_N)
    load_checkpoint_for_testing(gen_D_checkpoint, gen_D)   

    dataset = ppd.DayNightDataset(
        root_day=DIR + "/day",
        root_night=DIR + "/night",
        size=SIZE,
        transform=transforms,
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

        # kid requires uint8
        day_img_uint8 = (day_img * 255).to(torch.uint8)
        night_img_uint8 = (night_img * 255).to(torch.uint8)
        fake_night_uint8 = (fake_night * 255).to(torch.uint8)
        fake_day_uint8 = (fake_day * 255).to(torch.uint8)

        day_images.append(day_img_uint8)
        night_images.append(night_img_uint8)
        generated_night_images.append(fake_night_uint8)
        generated_day_images.append(fake_day_uint8)
        
    _ = torch.manual_seed(42)

    kid_D2N = KernelInceptionDistance(subsets=4, subset_size=40).to(DEVICE)
    kid_N2D = KernelInceptionDistance(subsets=4, subset_size=40).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        kid_D2N.update(night, real=True)
        kid_D2N.update(gen_night, real=False)
        
    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        kid_N2D.update(day, real=True)
        kid_N2D.update(gen_day, real=False)

    kid_D2N_mean, kid_D2N_std = kid_D2N.compute()
    kid_N2D_mean, kid_N2D_std = kid_N2D.compute()

    fid_D2N = FrechetInceptionDistance(feature=64).to(DEVICE)
    fid_N2D = FrechetInceptionDistance(feature=64).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        fid_D2N.update(night, real=True)
        fid_D2N.update(gen_night, real=False)
        
    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        fid_N2D.update(day, real=True)
        fid_N2D.update(gen_day, real=False)

    fid_D2N = fid_D2N.compute()
    fid_N2D = fid_N2D.compute()
    
    print(f"Kernel Inception Distance Mean: {kid_D2N_mean.item()}, Standard Deviation: {kid_D2N_std.item()}")
    print(f"Kernel Inception Distance Mean: {kid_N2D_mean.item()}, Standard Deviation: {kid_N2D_std.item()}")

    print(f"Frechet Inception Distance: {fid_D2N.item()}")
    print(f"Frechet Inception Distance: {fid_N2D.item()}")
    
if __name__ == "__main__":
    main()