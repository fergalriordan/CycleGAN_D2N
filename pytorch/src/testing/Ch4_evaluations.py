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
import pandas as pd

from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing import preprocess_data as ppd

from models import unet_resnet18_encoder as un_res
from models import resnet18_encoder as resn_enc
from models import resnet18_decoder as resn_dec
from models import timestamped_resnet_decoder as time_resn_dec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

DIR = "../data/train"

SIZE = 224 # generate images with the same dimensions as the training data for the inception model

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
        "--basic_path", type=str, help="Path to the basic U-Net generator checkpoints"
    )
    parser.add_argument(
        "--sharing_path", type=str, help="Path to the sharing U-Net generator checkpoints"
    )
    parser.add_argument(
        "--single_path", type=str, help="Path to the single generator checkpoint"
    )
    return parser.parse_args()

def generate_images(gen_N, gen_D, loader):
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
    
    return day_images, night_images, generated_night_images, generated_day_images

def generate_images_from_timestamps(gen, loader):
    day_images = []
    night_images = []
    generated_night_images = []
    generated_day_images = []

    for idx, (night_img, day_img) in enumerate(loader):
        day_img = day_img.to(DEVICE)
        night_img = night_img.to(DEVICE)

        timestamp0 = torch.tensor([0.0], device=DEVICE).float() 
        timestamp100 = torch.tensor([1.0], device=DEVICE).float()

        with torch.no_grad():
            fake_night = gen(day_img, timestamp100)
            fake_day = gen(night_img, timestamp0)
            
        day_images.append(day_img)
        night_images.append(night_img)
        generated_night_images.append(fake_night)
        generated_day_images.append(fake_day)
    
    return day_images, night_images, generated_night_images, generated_day_images


def compute_kid_scores(day_images, night_images, generated_night_images, generated_day_images):
    kid_D2N = KernelInceptionDistance(subsets=6, subset_size=50, normalize=True).to(DEVICE)
    kid_N2D = KernelInceptionDistance(subsets=10, subset_size=50, normalize=True).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        kid_D2N.update(night, real=True)
        kid_D2N.update(gen_night, real=False)
        
    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        kid_N2D.update(day, real=True)
        kid_N2D.update(gen_day, real=False)

    kid_N_mean, kid_N_std = kid_D2N.compute()
    kid_D_mean, kid_D_std = kid_N2D.compute()

    print(f"Kernel Inception Distance Mean (D2N): {kid_N_mean.item()}, Standard Deviation: {kid_N_std.item()}")
    print(f"Kernel Inception Distance Mean (N2D): {kid_D_mean.item()}, Standard Deviation: {kid_D_std.item()}")

    return kid_N_mean, kid_N_std, kid_D_mean, kid_D_std

def compute_fid_scores(day_images, night_images, generated_night_images, generated_day_images):
    fid_D2N = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)
    fid_N2D = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    for idx, (night, gen_night) in enumerate(zip(night_images, generated_night_images)):
        fid_D2N.update(night, real=True)
        fid_D2N.update(gen_night, real=False)

    for idx, (day, gen_day) in enumerate(zip(day_images, generated_day_images)):
        fid_N2D.update(day, real=True)
        fid_N2D.update(gen_day, real=False)

    fid_N_mean = fid_D2N.compute()
    fid_D_mean = fid_N2D.compute()    

    print(f"Frechet Inception Distance Mean (D2N): {fid_N_mean.item()}")
    print(f"Frechet Inception Distance Mean (N2D): {fid_D_mean.item()}")

    return fid_N_mean, fid_D_mean

def calculate_metrics(gen_N, gen_D, loader):
    day_images, night_images, generated_night_images, generated_day_images = generate_images(gen_N, gen_D, loader)
    kid_N_mean, kid_N_std, kid_D_mean, kid_D_std = compute_kid_scores(day_images, night_images, generated_night_images, generated_day_images)
    fid_N_mean, fid_D_mean = compute_fid_scores(day_images, night_images, generated_night_images, generated_day_images)

    return kid_N_mean, kid_N_std, kid_D_mean, kid_D_std, fid_N_mean, fid_D_mean

def calculate_metrics_timestamped_generator(gen, loader):
    day_images, night_images, generated_night_images, generated_day_images = generate_images_from_timestamps(gen, loader)
    kid_N_mean, kid_N_std, kid_D_mean, kid_D_std = compute_kid_scores(day_images, night_images, generated_night_images, generated_day_images)
    fid_N_mean, fid_D_mean = compute_fid_scores(day_images, night_images, generated_night_images, generated_day_images)

    return kid_N_mean, kid_N_std, kid_D_mean, kid_D_std, fid_N_mean, fid_D_mean


def evaluate_model(model_path, gen_N, gen_D, loader):
    results_list = []
    
    for epoch in range(10, 101, 10):
        print(f"Epoch: {epoch}")
        gen_N_checkpoint = model_path + f"epoch_{epoch}/gen_n_{epoch}.pth.tar"
        gen_D_checkpoint = model_path + f"epoch_{epoch}/gen_d_{epoch}.pth.tar"    

        load_checkpoint_for_testing(gen_N_checkpoint, gen_N)
        load_checkpoint_for_testing(gen_D_checkpoint, gen_D)

        kid_N_mean, kid_N_std, kid_D_mean, kid_D_std, fid_N_mean, fid_D_mean = calculate_metrics(gen_N, gen_D, loader)
        
        results_list.append({
            "Epoch": epoch,
            "KID_N_mean": kid_N_mean.item(),
            "KID_N_std": kid_N_std.item(),
            "KID_D_mean": kid_D_mean.item(),
            "KID_D_std": kid_D_std.item(),
            "FID_N_mean": fid_N_mean.item(),
            "FID_D_mean": fid_D_mean.item(),
        })
        
    return pd.DataFrame(results_list)

def evaluate_timestamped_model(model_path, gen, loader):
    results_list = []
    
    for epoch in range(10, 101, 10):
        print(f"Epoch: {epoch}")
        gen_N_checkpoint = model_path + f"epoch_{epoch}/gen_{epoch}.pth.tar"  

        load_checkpoint_for_testing(gen_N_checkpoint, gen)

        kid_N_mean, kid_N_std, kid_D_mean, kid_D_std, fid_N_mean, fid_D_mean = calculate_metrics_timestamped_generator(gen, loader)
        
        results_list.append({
            "Epoch": epoch,
            "KID_N_mean": kid_N_mean.item(),
            "KID_N_std": kid_N_std.item(),
            "KID_D_mean": kid_D_mean.item(),
            "KID_D_std": kid_D_std.item(),
            "FID_N_mean": fid_N_mean.item(),
            "FID_D_mean": fid_D_mean.item(),
        })
        
    return pd.DataFrame(results_list)

def main():
    args = parse_arguments()

    day_transforms = ppd.set_val_transforms(day_means, day_stds)
    night_transforms = ppd.set_val_transforms(night_means, night_stds)

    dataset = ppd.DayNightDataset(
        root_day=DIR + "/day",
        root_night=DIR + "/night",
        size=SIZE,
        day_transform=day_transforms,
        night_transform=night_transforms,
    )

    loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
    )

    # ResNet generators
    basic_gen_N = un_res.UnetResNet18(output_channels=3).to(DEVICE)
    basic_gen_D = un_res.UnetResNet18(output_channels=3).to(DEVICE)

    # Sharing generators
    encoder = resn_enc.ResNet18Encoder().to(DEVICE)
    sharing_gen_N = resn_dec.ResNet18Decoder(encoder, output_channels=3).to(DEVICE)
    sharing_gen_D = resn_dec.ResNet18Decoder(encoder, output_channels=3).to(DEVICE)

    # Timestamped generator
    encoder = resn_enc.ResNet18Encoder().to(DEVICE)
    single_gen = time_resn_dec.TimestampedResNet18Decoder(encoder, 640, 3).to(DEVICE)

    basic_path = args.basic_path
    sharing_path = args.sharing_path
    single_path = args.single_path

    _ = torch.manual_seed(42)

    print("Evaluating Basic...")
    DnCNN_results = evaluate_model(basic_path, basic_gen_N, basic_gen_D, loader)
    DnCNN_results.to_csv('basic.csv', index=False)
    print('Basic results saved to basic.csv')
    print("Evaluating Sharing...")
    UNet_results = evaluate_model(sharing_path, sharing_gen_N, sharing_gen_D, loader)
    UNet_results.to_csv('sharing.csv', index=False)
    print('Sharing results saved to sharing.csv')
    print("Evaluating Timestamped...")
    ResNet_results = evaluate_timestamped_model(single_path, single_gen, loader)
    ResNet_results.to_csv('timestamped.csv', index=False)
    print('Timestamped results saved to timestamped.csv')

if __name__ == "__main__":
    seed_everything()
    main()