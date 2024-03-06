import torch 
from torch.utils.data import DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image

import sys
import os
import argparse

from preprocessing import preprocess_data as ppd

from models import generator as gen
from models import encoder as enc
from models import sharing_generator as sh_gen

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

VAL_DIR = "../data/val"
TEST_SIZE = 2048

val_transforms = A.Compose(
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

    val_dataset = ppd.DayNightDataset(
        root_day=VAL_DIR + "/day",
        root_night=VAL_DIR + "/night",
        size=TEST_SIZE,
        transform=val_transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        pin_memory=True,
    )

    for idx, (night_img, day_img) in enumerate(val_loader):
        day_img = day_img.to(DEVICE)
        night_img = night_img.to(DEVICE)

        with torch.no_grad():
            fake_night = gen_N(day_img)
            fake_day = gen_D(night_img)

        output_directory = f"../outputs/testing/{model_type}_{epoch}/"

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
              
        save_image(fake_night*0.5+0.5, os.path.join(output_directory, f"night_{idx}.png"))
        save_image(fake_day*0.5+0.5, os.path.join(output_directory, f"day_{idx}.png"))

if __name__ == "__main__":
    main()