import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from torchsummary import summary

import sys, argparse, random, os, numpy as np
from tqdm import tqdm

# Fix the path so the other scripts can be imported 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Preprocessing
from preprocessing import preprocess_data as ppd

# Models
from models import discriminator as disc
from models import unet_encoder as un_enc
from models import timestamped_unet_decoder as time_un_dec

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

# Image directories
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"
TIMELAPSE_TRAIN_DIR = "../data/timelapse_training_data"
TIMELAPSE_VAL_DIR = "../data/timelapse_val_data"

# Hyperparameters
BATCH_SIZE = 1
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.5 

NUM_EPOCHS_MAIN_TRAINING = 50 # day <=> night (big dataset)
TRAINING_CONFIG = [ # configurations for timelapse traiingin
    (3, 0.0, 1.0), 
    (3, 0.0, 0.5), 
    (3, 0.5, 1.0),
    (3, 0.0, 1.0),
    (3, 0.0, 0.5),
    (3, 0.5, 1.0),
    (3, 0.0, 1.0),
    (3, 0.0, 0.5),
    (3, 0.5, 1.0),
    (3, 0.0, 1.0)
    ]

# Dimensions of validation images is fixed, whereas training image dimensions depend on whether a pre-trained encoder is used (224x224) or not (256x256)
TEST_SIZE = 1028 

NUM_WORKERS = 2

CHECKPOINT_INCREMENT = 5 # generate test outputs and save model checkpoints at epoch increments of this number

# Checkpoint names
CHECKPOINT_GENERATOR = "gen"
CHECKPOINT_DISCRIMINATOR_D = "disc_0"
CHECKPOINT_DISCRIMINATOR_N = "disc_100"
CHECKPOINT_DISCRIMINATOR_DUSK = "disc_50"
CHECKPOINT_DISCRIMINATOR_LATE_DUSK = "disc_75"
CHECKPOINT_DISCRIMINATOR_EARLY_DUSK = "disc_25"

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

def get_loaders(train_dir1, train_dir2, val_dir1, val_dir2, training_image_size, validation_image_size, transforms, batch_size, num_workers):

    dataset = ppd.DayNightDataset(
        root_day=train_dir1,
        root_night=train_dir2,
        size=training_image_size,
        transform=transforms,
    )
    val_dataset = ppd.DayNightDataset(
        root_day=val_dir1,
        root_night=val_dir2,
        size=validation_image_size,
        transform=ppd.val_transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    return loader, val_loader

def train_fn(
    disc_D, disc_N, gen, timestamp0, timestamp1, loader, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch
):
    Day_reals = 0
    Day_fakes = 0
    loop = tqdm(loader, leave=True) # progress bar

    day_timestamp = torch.tensor([timestamp0], device=DEVICE).float()  # 0 = day
    night_timestamp = torch.tensor([timestamp1], device=DEVICE).float() # 1 = night

    for idx, (night, day) in enumerate(loop):
        # read one batch of images from the data loader and move them to the GPU
        # in my case, the batch size will be set to 1 in the data loader, meaning only one day image and one night image will be processed at a time
        night = night.to(DEVICE)
        day = day.to(DEVICE) 

        # Discriminator training (disc_D and disc_N)
        with torch.cuda.amp.autocast():
            # Two discriminators are trained in parallel
            # Process for a general discriminator disc_A is as follows:
            # 1. Generate a fake image A_fake from real image B
            # 2. Make predictions on real image A and fake image A_fake
            # 3. Calculate the MSE between the prediction made for A and 1 (the correct prediction)
            # 4. Calculate the MSE between the prediction made for A_fake and 0 (the correct prediction)
            # 5. Sum the losses

            fake_day = gen(night, day_timestamp)               
            D_Day_real = disc_D(day) 
            D_Day_fake = disc_D(fake_day.detach())
            Day_reals += D_Day_real.mean().item() # keep track of the cumulated predictions on real day images
            Day_fakes += D_Day_fake.mean().item() # do the same for predictions on fake day images - rolling avg printed for monitoring purposes
            D_Day_real_loss = mse(D_Day_real, torch.ones_like(D_Day_real)) 
            D_Day_fake_loss = mse(D_Day_fake, torch.zeros_like(D_Day_fake)) 
            D_Day_loss = D_Day_real_loss + D_Day_fake_loss 

            fake_night = gen(day, night_timestamp) 
            D_Night_real = disc_N(night) 
            D_Night_fake = disc_N(fake_night.detach()) 
            D_Night_real_loss = mse(D_Night_real, torch.ones_like(D_Night_real)) 
            D_Night_fake_loss = mse(D_Night_fake, torch.zeros_like(D_Night_fake)) 
            D_Night_loss = D_Night_real_loss + D_Night_fake_loss 

            D_loss = (D_Day_loss + D_Night_loss) / 2 # compute average of disc_D loss and disc_N loss for overall discriminator loss

        opt_disc.zero_grad() # reset the gradients of the discriminator optimizer to avoid accumulation from previous iteration
        d_scaler.scale(D_loss).backward() # scale the loss to avoid gradient underflow during backpropagation (mixed precision training might otherwise lead to some values being set to zero)
        d_scaler.step(opt_disc) # unscale gradients back to original values, then use them to update the discriminator weights
        d_scaler.update() # adjust gradient scale factor for next iteration

        # Generator training 
        with torch.cuda.amp.autocast():
            # Adversarial loss terms:
            # MSE between discriminator prediction on the fake images and an incorrect discriminator prediction
            D_Day_fake = disc_D(fake_day) 
            D_Night_fake = disc_N(fake_night)
            loss_G_Day = mse(D_Day_fake, torch.ones_like(D_Day_fake))
            loss_G_Night = mse(D_Night_fake, torch.ones_like(D_Night_fake))

            # Cycle losses:
            # L1 loss between the original image and the reconstructed image after a full cycle
            cycle_night = gen(fake_day, night_timestamp)
            cycle_day = gen(fake_night, day_timestamp)
            cycle_night_loss = l1(night, cycle_night)
            cycle_day_loss = l1(day, cycle_day)

            # Identity losses
            # L1 loss between the original image and the reconstructed image after an identity mapping 
            # Encourage generators to make no change if an image already in their target domain is passed as an input
            identity_night = gen(night, night_timestamp)
            identity_day = gen(day, day_timestamp)
            identity_night_loss = l1(night, identity_night)
            identity_day_loss = l1(day, identity_day)

            # Calculate basic generator loss (before any optional extra terms)
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

        # Every 100 images, save a low-res snapshot (for monitoring purposes)
        if idx % 100 == 0:
            save_image(fake_day * 0.5 + 0.5, f"../outputs/training/snapshots/day_{idx}.png")
            save_image(fake_night * 0.5 + 0.5, f"../outputs/training/snapshots/night_{idx}.png")

        # print rolling average of predictions on real and fake day images for monitoring purposes
        # (if D_real approaches 1 and D_fake approaches 0, the discriminator is outperforming the generator)
        loop.set_postfix(D_real=Day_reals / (idx + 1), D_fake=Day_fakes / (idx + 1))

    # Validation phase after each epoch (only performed if current epoch is divisible by the CHECKPOINT_INCREMENT)
    if epoch % CHECKPOINT_INCREMENT == 0:
      gen.eval()
      with torch.no_grad():
          for val_idx, (val_night, val_day) in enumerate(val_loader):
              val_night = val_night.to(DEVICE)
              val_day = val_day.to(DEVICE)
              
              timestamp0 = torch.tensor([0.0], device=DEVICE).float()  
              timestamp50 = torch.tensor([0.5], device=DEVICE).float() 
              timestamp100 = torch.tensor([1.0], device=DEVICE).float()

              # Generate images using generators
              val_0_from_0 = gen(val_day, timestamp0)
              val_50_from_0 = gen(val_day, timestamp50)
              val_100_from_0 = gen(val_day, timestamp100)

              val_100_from_100 = gen(val_night, timestamp100)
              val_50_from_100 = gen(val_night, timestamp50)
              val_0_from_100 = gen(val_night, timestamp0)
              #val_fake_day = gen(val_night, day_timestamp)
              #val_fake_night = gen(val_day, night_timestamp)

              output_directory = f"../outputs/training/outputs/epoch_{epoch}/"

              if not os.path.exists(output_directory):
                os.makedirs(output_directory)
              
              # Save the images
              save_image(val_0_from_0 * 0.5 + 0.5, os.path.join(output_directory, f"day-to-day_{val_idx}.png"))
              save_image(val_50_from_0 * 0.5 + 0.5, os.path.join(output_directory, f"day-to-dusk_{val_idx}.png"))
              save_image(val_100_from_0 * 0.5 + 0.5, os.path.join(output_directory, f"day-to-night_{val_idx}.png"))
              save_image(val_100_from_100 * 0.5 + 0.5, os.path.join(output_directory, f"night-to-night_{val_idx}.png"))
              save_image(val_50_from_100 * 0.5 + 0.5, os.path.join(output_directory, f"night-to-dusk_{val_idx}.png"))
              save_image(val_0_from_100 * 0.5 + 0.5, os.path.join(output_directory, f"night-to-day_{val_idx}.png"))
              
def parse_arguments():

    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('--load_model', type=str, default='n',
                        help='Path to a checkpoint folder, e.g. "../outputs/training/checkpoints/epoch_200/" (default: n)')
    parser.add_argument('--loaded_epoch', type=str, default='n',
                        help='Number of epochs of training that have already been performed for the loaded model (default: n)')
    parser.add_argument('--disc_learning_rate', type=float, default=2e-4, 
                        help='Discriminator learning rate (default: 2e-4)')
    parser.add_argument('--gen_learning_rate', type=float, default=2e-4, 
                        help='Generator learning rate (default: 2e-4)')
    
    return parser.parse_args()

def main():
    
    args = parse_arguments()

    training_image_size = 256

    disc_D = disc.Discriminator(in_channels=3).to(DEVICE) 
    disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
    encoder = un_enc.UNet_Encoder(input_channel=3).to(DEVICE)
    gen = time_un_dec.Timestamped_UNet_Decoder(encoder, 512, 3).to(DEVICE)

    # extra discriminators for timelapse training
    disc_Dusk = disc.Discriminator(in_channels=3).to(DEVICE)
    #disc_Early_Dusk = disc.Discriminator(in_channels=3).to(DEVICE)
    #disc_Late_Dusk = disc.Discriminator(in_channels=3).to(DEVICE)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = optim.Adam(
        list(disc_D.parameters()) + list(disc_N.parameters()), lr=args.disc_learning_rate, betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen.parameters()), lr=args.gen_learning_rate, betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    offset = 0 # no epoch offset if no loaded model (training from scratch), otherwise the loaded epoch must be added to the current epoch

    if args.load_model != 'n' and args.loaded_epoch == 'n':
        print("Invalid model loading arguments. You must specify the current epoch of the loaded checkpoint.")
        return

    if args.load_model != 'n':
        offset = int(args.loaded_epoch)

        gen_checkpoint = args.load_model + f"{CHECKPOINT_GENERATOR}_{args.loaded_epoch}.pth.tar"
        disc_D_checkpoint = args.load_model + f"{CHECKPOINT_DISCRIMINATOR_D}_{args.loaded_epoch}.pth.tar"
        disc_N_checkpoint = args.load_model + f"{CHECKPOINT_DISCRIMINATOR_N}_{args.loaded_epoch}.pth.tar"

        load_checkpoint(gen_checkpoint, gen, opt_gen, args.gen_learning_rate)
        load_checkpoint(disc_D_checkpoint, disc_D, opt_disc, args.disc_learning_rate)
        load_checkpoint(disc_N_checkpoint, disc_N, opt_disc, args.disc_learning_rate)
    
    transforms = ppd.set_training_transforms(training_image_size)

    loader0, val_loader0 = get_loaders(
        TRAIN_DIR + "/day", 
        TRAIN_DIR + "/night", 
        VAL_DIR + "/day", 
        VAL_DIR + "/night", 
        training_image_size,
        TEST_SIZE, 
        transforms, 
        BATCH_SIZE, 
        NUM_WORKERS
    )

    loader1, val_loader1 = get_loaders(
        TIMELAPSE_TRAIN_DIR + "/time_0",
        TIMELAPSE_TRAIN_DIR + "/time_100",
        TIMELAPSE_VAL_DIR + "/time_0",
        TIMELAPSE_VAL_DIR + "/time_100",
        training_image_size,
        TEST_SIZE,
        transforms,
        BATCH_SIZE,
        NUM_WORKERS
    )

    loader2, val_loader2 = get_loaders(
        TIMELAPSE_TRAIN_DIR + "/time_0",
        TIMELAPSE_TRAIN_DIR + "/time_50",
        TIMELAPSE_VAL_DIR + "/time_0",
        TIMELAPSE_VAL_DIR + "/time_50",
        training_image_size,
        TEST_SIZE,
        transforms,
        BATCH_SIZE,
        NUM_WORKERS
    )

    loader3, val_loader3 = get_loaders(
        TIMELAPSE_TRAIN_DIR + "/time_50",
        TIMELAPSE_TRAIN_DIR + "/time_100",
        TIMELAPSE_VAL_DIR + "/time_50",
        TIMELAPSE_VAL_DIR + "/time_100",
        training_image_size,
        TEST_SIZE,
        transforms,
        BATCH_SIZE,
        NUM_WORKERS
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Primary Training
    print("STARTING MAIN TRAINING LOOP...")

    for epoch in range(NUM_EPOCHS_MAIN_TRAINING):
        print('Epoch: ', (epoch+1), '/', NUM_EPOCHS_MAIN_TRAINING)

        timestamp0 = 0.0
        timestamp1 = 1.0

        train_fn(
            disc_D,
            disc_N,
            gen,
            timestamp0,
            timestamp1,
            loader0,
            val_loader0,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
            (epoch+1)
        )
            
        if (epoch+1) % CHECKPOINT_INCREMENT == 0:
            epoch_folder = f"../outputs/training/checkpoints/epoch_{epoch+1+offset}/"

            if not os.path.exists(epoch_folder):
                os.makedirs(epoch_folder)

            save_checkpoint(gen, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR}_{epoch+1+offset}.pth.tar"))
            save_checkpoint(disc_D, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_D}_{epoch+1+offset}.pth.tar"))
            save_checkpoint(disc_N, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_N}_{epoch+1+offset}.pth.tar"))
    
    offset += NUM_EPOCHS_MAIN_TRAINING

    # Timelapse Training
    print("STARTING TIMELAPSE TRAINING...")

    for phase, (num_epochs, timestamp0, timestamp1) in enumerate(TRAINING_CONFIG, start=1):
        print('Phase: ', phase, '/', len(TRAINING_CONFIG))
        print(timestamp0, '<=>', timestamp1)

        if timestamp0 == 0.0 and timestamp1 == 1.0:
            loader, val_loader = loader1, val_loader1
            disc_x, disc_y = disc_D, disc_N
            disc_x_checkpoint_name, disc_y_checkpoint_name = CHECKPOINT_DISCRIMINATOR_D, CHECKPOINT_DISCRIMINATOR_N
        elif timestamp0 == 0.0 and timestamp1 == 0.5:
            loader, val_loader = loader2, val_loader2
            disc_x, disc_y = disc_D, disc_Dusk
            disc_x_checkpoint_name, disc_y_checkpoint_name = CHECKPOINT_DISCRIMINATOR_D, CHECKPOINT_DISCRIMINATOR_DUSK
        elif timestamp0 == 0.5 and timestamp1 == 1.0:
            loader, val_loader = loader3, val_loader3
            disc_x, disc_y = disc_Dusk, disc_N
            disc_x_checkpoint_name, disc_y_checkpoint_name = CHECKPOINT_DISCRIMINATOR_DUSK, CHECKPOINT_DISCRIMINATOR_N
        else : 
            raise ValueError("Invalid timestamp configuration")
        
        for epoch in range(num_epochs):
            print('Epoch: ', (epoch+1), '/', num_epochs)

            train_fn(
                disc_x,
                disc_y,
                gen,
                timestamp0,
                timestamp1,
                loader,
                val_loader,
                opt_disc,
                opt_gen,
                L1,
                mse,
                d_scaler,
                g_scaler,
                (epoch+1+offset)
            )

            if (epoch+1) % CHECKPOINT_INCREMENT == 0:
                epoch_folder = f"../outputs/training/checkpoints/epoch_{epoch+1+offset}/"

                if not os.path.exists(epoch_folder):
                    os.makedirs(epoch_folder)
                
                save_checkpoint(gen, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR}_{epoch+1+offset}.pth.tar"))
                save_checkpoint(disc_x, opt_disc, filename=os.path.join(epoch_folder, f"{disc_x_checkpoint_name}_{epoch+1+offset}.pth.tar"))
                save_checkpoint(disc_y, opt_disc, filename=os.path.join(epoch_folder, f"{disc_y_checkpoint_name}_{epoch+1+offset}.pth.tar"))
        
        offset += num_epochs


if __name__ == "__main__":
    seed_everything(seed=123) # make the training as reproducible as possible
    main()
