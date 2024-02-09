import torch
import torch.nn as nn
from torch.utils.data import DataLoader#
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
import random, os, numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import generator, discriminator and preprocessing files
from models import generator as gen
from models import discriminator as disc
from models import preprocess_data as ppd
from models import encoder as enc
from models import sharing_generator as sh_gen

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"
TEST_DIR = "../data/val"

# Hyperparameters
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.5 # loss weight for identity loss (0.5 in TensorFlow implementation)
LAMBDA_CYCLE = 10
LAMBDA_MID = 5
NUM_WORKERS = 4
NUM_EPOCHS = 200
TRAINING_SIZE = 256
TEST_SIZE = 1028

# Training/Testing?
LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_INCREMENT = 10 # generate test outputs and save model checkpoints at epoch increments of this number

CHECKPOINT_GENERATOR_D = "gen_d"
CHECKPOINT_GENERATOR_N = "gen_n"
CHECKPOINT_DISCRIMINATOR_D = "disc_d"
CHECKPOINT_DISCRIMINATOR_N = "disc_n"

transforms = A.Compose(
    [
        #A.Resize(width=256, height=256),
        A.RandomResizedCrop(height=256, width=256, scale=(0.1, 1.0), ratio=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

val_transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)

def save_checkpoint(model, optimizer, filename="../outputs/training/checkpoints/checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_fn(
    disc_D, disc_N, gen_N, gen_D, loader, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, encoder=None
):
    Day_reals = 0
    Day_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (night, day) in enumerate(loop):
        night = night.to(DEVICE)
        day = day.to(DEVICE)

        # Train discriminators D and N
        with torch.cuda.amp.autocast():
            fake_day = gen_D(night)
            D_Day_real = disc_D(day)
            D_Day_fake = disc_D(fake_day.detach())
            Day_reals += D_Day_real.mean().item()
            Day_fakes += D_Day_fake.mean().item()
            D_Day_real_loss = mse(D_Day_real, torch.ones_like(D_Day_real))
            D_Day_fake_loss = mse(D_Day_fake, torch.zeros_like(D_Day_fake))
            D_Day_loss = D_Day_real_loss + D_Day_fake_loss

            fake_night = gen_N(day)
            D_Night_real = disc_N(night)
            D_Night_fake = disc_N(fake_night.detach())
            D_Night_real_loss = mse(D_Night_real, torch.ones_like(D_Night_real))
            D_Night_fake_loss = mse(D_Night_fake, torch.zeros_like(D_Night_fake))
            D_Night_loss = D_Night_real_loss + D_Night_fake_loss

            D_loss = (D_Day_loss + D_Night_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators D and N
        with torch.cuda.amp.autocast():
            # adversarial losses
            D_Day_fake = disc_D(fake_day)
            D_Night_fake = disc_N(fake_night)
            loss_G_Day = mse(D_Day_fake, torch.ones_like(D_Day_fake))
            loss_G_Night = mse(D_Night_fake, torch.ones_like(D_Night_fake))

            # cycle losses
            cycle_night = gen_N(fake_day)
            cycle_day = gen_D(fake_night)
            cycle_night_loss = l1(night, cycle_night)
            cycle_day_loss = l1(day, cycle_day)

            # identity losses
            identity_night = gen_N(night)
            identity_day = gen_D(day)
            identity_night_loss = l1(night, identity_night)
            identity_day_loss = l1(day, identity_day)

            # if an encoder is passed, include mid-cycle loss term
            if encoder is not None:
                mid_cycle_night = encoder(day)
                mid_cycle_day = encoder(night)
                mid_cycle_loss = l1(mid_cycle_day, mid_cycle_night)

                # total loss
                G_loss = (
                    loss_G_Night
                    + loss_G_Day
                    + cycle_night_loss * LAMBDA_CYCLE
                    + cycle_day_loss * LAMBDA_CYCLE
                    + identity_day_loss * LAMBDA_IDENTITY
                    + identity_night_loss * LAMBDA_IDENTITY
                    + mid_cycle_loss * LAMBDA_MID
                )
            
            # otherwise, no mid-cycle loss
            else :
                # total loss
                G_loss = (
                    loss_G_Night
                    + loss_G_Day
                    + cycle_night_loss * LAMBDA_CYCLE
                    + cycle_day_loss * LAMBDA_CYCLE
                    + identity_day_loss * LAMBDA_IDENTITY
                    + identity_night_loss * LAMBDA_IDENTITY
                )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 100 == 0:
            save_image(fake_day * 0.5 + 0.5, f"../outputs/training/snapshots/day_{idx}.png")
            save_image(fake_night * 0.5 + 0.5, f"../outputs/training/snapshots/night_{idx}.png")

        loop.set_postfix(D_real=Day_reals / (idx + 1), D_fake=Day_fakes / (idx + 1))


    # Validation phase after each epoch
    if epoch % CHECKPOINT_INCREMENT == 0:
      gen_N.eval()
      gen_D.eval()
      with torch.no_grad():
          for val_idx, (val_night, val_day) in enumerate(val_loader):
              val_night = val_night.to(DEVICE)
              val_day = val_day.to(DEVICE)

              # Generate images using generators
              val_fake_day = gen_D(val_night)
              val_fake_night = gen_N(val_day)

              output_directory = f"../outputs/training/outputs/epoch_{epoch}/"

              if not os.path.exists(output_directory):
                os.makedirs(output_directory)
              
              # Save the images
              save_image(val_fake_day * 0.5 + 0.5, os.path.join(output_directory, f"day_{val_idx}.png"))
              save_image(val_fake_night * 0.5 + 0.5, os.path.join(output_directory, f"night_{val_idx}.png"))
              
def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--generator_type', type=str, choices=['simple', 'sharing'], default='simple',
                        help='Type of generator to use: simple or sharing')
    parser.add_argument('--mid_cycle_loss', type=str, choices=['y', 'n'], default='n',
                        help='Include a mid-cycle loss term to constrain the representations generated by the shared encoder: y or n')
    parser.add_argument('--load_model', type=str, default='n',
                        help='Path to a folder containing model checkpoints to load a pretrained model, e.g. "../outputs/training/checkpoints/epoch_200/" (default: n)')
    parser.add_argument('--loaded_epoch', type=str, default='n',
                        help='Epoch of a pretrained model to load (default: n)')
    return parser.parse_args()

def main():
    
    args = parse_arguments()
    
    generator_type = args.generator_type
    if generator_type not in ['simple', 'sharing']:
        print("Invalid generator type. Choose either 'simple' or 'sharing'.")
        return
    
    mid_cycle_loss = args.mid_cycle_loss
    if mid_cycle_loss not in ['y', 'n']:
        print("Invalid mid-cycle loss argument. Choose either 'y' or 'n'.")
        return
    
    if generator_type == 'simple':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        gen_N = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
        gen_D = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
    elif generator_type == 'sharing':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        encoder = enc.Encoder(img_channels=3, num_features=64).to(DEVICE)
        gen_N = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)
        gen_D = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = optim.Adam(
        list(disc_D.parameters()) + list(disc_N.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_N.parameters()) + list(gen_D.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    load_model = args.load_model
    loaded_epoch = args.loaded_epoch
    if load_model != 'n' and loaded_epoch == 'n':
        print("Invalid model loading arguments. You must specify the epoch of the model to load (e.g. --load_model=\"../outputs/training/checkpoints/epoch_200/\" --loaded_epoch=200).")
        return
    
    E = 0 # variable for naming of checkpoints - will be zero if no loaded model (therefore checkpoints will start from epoch_0), otherwise, the loaded epoch is added

    #if LOAD_MODEL:
    if load_model != 'n':
        # load the checkpoints stored at the given directory

        E = loaded_epoch

        gen_D_checkpoint = load_model + f"{CHECKPOINT_GENERATOR_D}_{loaded_epoch}.pth.tar"
        gen_N_checkpoint = load_model + f"{CHECKPOINT_GENERATOR_N}_{loaded_epoch}.pth.tar"
        disc_D_checkpoint = load_model + f"{CHECKPOINT_DISCRIMINATOR_D}_{loaded_epoch}.pth.tar"
        disc_N_checkpoint = load_model + f"{CHECKPOINT_DISCRIMINATOR_N}_{loaded_epoch}.pth.tar"

        load_checkpoint(
            gen_D_checkpoint,
            gen_D,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            gen_N_checkpoint,
            gen_N,
            opt_gen,
            LEARNING_RATE,
        )
        load_checkpoint(
            disc_D_checkpoint,
            disc_D,
            opt_disc,
            LEARNING_RATE,
        )
        load_checkpoint(
            disc_N_checkpoint,
            disc_N,
            opt_disc,
            LEARNING_RATE,
        )

    dataset = ppd.DayNightDataset(
        root_day=TRAIN_DIR + "/day",
        root_night=TRAIN_DIR + "/night",
        size=TRAINING_SIZE,
        transform=transforms,
    )
    val_dataset = ppd.DayNightDataset(
        root_day=VAL_DIR + "/day",
        root_night=VAL_DIR + "/night",
        size=TEST_SIZE,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print('Epoch: ', (epoch+1), '/', NUM_EPOCHS)

        if mid_cycle_loss == 'n':
            train_fn(
                disc_D,
                disc_N,
                gen_N,
                gen_D,
                loader,
                val_loader,
                opt_disc,
                opt_gen,
                L1,
                mse,
                d_scaler,
                g_scaler,
                (epoch+1),
            )
        
        elif mid_cycle_loss == 'y':
            train_fn(
                disc_D,
                disc_N,
                gen_N,
                gen_D,
                loader,
                val_loader,
                opt_disc,
                opt_gen,
                L1,
                mse,
                d_scaler,
                g_scaler,
                (epoch+1),
                encoder
            )
            
        if SAVE_MODEL and (epoch+1) % CHECKPOINT_INCREMENT == 0:
            epoch_folder = f"../outputs/training/checkpoints/epoch_{epoch+1}/"

            if not os.path.exists(epoch_folder):
                os.makedirs(epoch_folder)

            # checkpoints start from epoch 0 unless a model was loaded, in which case the loaded epoch (E) is added to the name (epoch_10 becomes epoch_210 if we loaded epoch_200 checkpoints) 
            save_checkpoint(gen_D, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR_D}_{epoch+1+E}.pth.tar"))
            save_checkpoint(gen_N, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR_N}_{epoch+1+E}.pth.tar"))
            save_checkpoint(disc_D, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_D}_{epoch+1+E}.pth.tar"))
            save_checkpoint(disc_N, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_N}_{epoch+1+E}.pth.tar"))


if __name__ == "__main__":
    main()
