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
from models import generator as gen
from models import discriminator as disc
from models import encoder as enc
from models import sharing_generator as sh_gen
from models import unet as un
from models import unet_encoder as un_enc
from models import unet_decoder as un_dec
from models import unet_resnet18_encoder as un_res
from models import resnet18_encoder as resn_enc
from models import resnet18_decoder as resn_dec

# Additional loss function
import lap_pyramid_loss as lpl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {DEVICE}")

# Image directories
TRAIN_DIR = "../data/train"
VAL_DIR = "../data/val"

# Hyperparameters
BATCH_SIZE = 1
LAMBDA_CYCLE = 10
LAMBDA_IDENTITY = 0.4
LAMBDA_MID = 0.5
LAMBDA_LAPLACIAN = 0.5
NUM_EPOCHS = 100

# Mean and standard deviation of day and night images
day_means = [0.5, 0.5, 0.5]
night_means = [0.5, 0.5, 0.5]
day_stds = [0.5, 0.5, 0.5]
night_stds = [0.5, 0.5, 0.5]

# Dimensions of validation images is fixed, whereas training image dimensions depend on whether a pre-trained encoder is used (224x224) or not (256x256)
TEST_SIZE = 1028 

NUM_WORKERS = 2

CHECKPOINT_INCREMENT = 5 # generate test outputs and save model checkpoints at epoch increments of this number

# Checkpoint names
CHECKPOINT_GENERATOR_D = "gen_d"
CHECKPOINT_GENERATOR_N = "gen_n"
CHECKPOINT_DISCRIMINATOR_D = "disc_d"
CHECKPOINT_DISCRIMINATOR_N = "disc_n"

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
    disc_D, disc_N, gen_N, gen_D, loader, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, laplace=False, encoder=None 
):
    Day_reals = 0
    Day_fakes = 0
    loop = tqdm(loader, leave=True) # progress bar

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
            # 3. Calculate the MSE between the prediction made for A and a tensor of 1s (the correct prediction)
            # 4. Calculate the MSE between the prediction made for A_fake and a tensor of 0s (the correct prediction)
            # 5. Sum the losses

            fake_day = gen_D(night)                       
            D_Day_real = disc_D(day) 
            D_Day_fake = disc_D(fake_day.detach())
            Day_reals += D_Day_real.mean().item() # keep track of the cumulated predictions on real day images
            Day_fakes += D_Day_fake.mean().item() # do the same for predictions on fake day images - rolling avg printed for monitoring purposes
            D_Day_real_loss = mse(D_Day_real, torch.ones_like(D_Day_real)) 
            D_Day_fake_loss = mse(D_Day_fake, torch.zeros_like(D_Day_fake)) 
            D_Day_loss = D_Day_real_loss + D_Day_fake_loss 

            fake_night = gen_N(day) 
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

        # Generator training (gen_D and gen_N)
        with torch.cuda.amp.autocast():
            # Adversarial loss terms:
            # MSE between discriminator prediction on the fake images and an incorrect discriminator prediction
            D_Day_fake = disc_D(fake_day) 
            D_Night_fake = disc_N(fake_night)
            loss_G_Day = mse(D_Day_fake, torch.ones_like(D_Day_fake))
            loss_G_Night = mse(D_Night_fake, torch.ones_like(D_Night_fake))

            # Cycle losses:
            # L1 loss between the original image and the reconstructed image after a full cycle
            cycle_night = gen_N(fake_day)
            cycle_day = gen_D(fake_night)
            cycle_night_loss = l1(night, cycle_night)
            cycle_day_loss = l1(day, cycle_day)

            # Identity losses
            # L1 loss between the original image and the reconstructed image after an identity mapping 
            # Encourage generators to make no change if an image already in their target domain is passed as an input
            identity_night = gen_N(night)
            identity_day = gen_D(day)
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
            
            # Optional loss term 1: Laplacian Pyramid Loss
            if laplace:
                laplace_pyramid_loss = lpl.LapLoss(device=DEVICE)
                G_loss = (G_loss 
                          + laplace_pyramid_loss(day, fake_night) * LAMBDA_LAPLACIAN
                          + laplace_pyramid_loss(night, fake_day) * LAMBDA_LAPLACIAN
                )

            # Optional loss term 2: Mid-cycle Loss (shared encoder architecture only)
            # Mid-cycle loss: L1 loss between latent space representaion of input after 1/4 cycle and 3/4 cycle
            # Encourage the encoder to produce a common latent representatoin regardless of input image's original domain
            if encoder is not None: # passing an encoder to the training function is a flag to include a mid-cycle loss term 
                mid_cycle_night1, _, _, _, _ = encoder(day) # u-net encoder returns a tuple (to facilitate skip connections to the upsampling decoder) but only the final latent representation is wanted for mid-cycle loss
                mid_cycle_night2, _, _, _, _ = encoder(fake_night)
                mid_cycle_night_loss = l1(mid_cycle_night1, mid_cycle_night2) 

                mid_cycle_day1, _, _, _, _ = encoder(night)
                mid_cycle_day2, _, _, _, _ = encoder(fake_day)
                mid_cycle_day_loss = l1(mid_cycle_day1, mid_cycle_day2)

                G_loss = (
                    G_loss
                    + mid_cycle_night_loss * LAMBDA_MID 
                    + mid_cycle_day_loss * LAMBDA_MID
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
      gen_N.eval()
      gen_D.eval()
      with torch.no_grad():
          for val_idx, (val_night, val_day) in enumerate(val_loader):
              val_night = val_night.to(DEVICE)
              val_day = val_day.to(DEVICE)

              # Generate images using generators
              val_fake_day = gen_D(val_night)
              val_fake_night = gen_N(val_day)

              output_directory = f"../outputs/training/full_images/epoch_{epoch}/"

              if not os.path.exists(output_directory):
                os.makedirs(output_directory)
              
              # Save the images
              save_image(val_fake_day * 0.5 + 0.5, os.path.join(output_directory, f"day_{val_idx}.png"))
              save_image(val_fake_night * 0.5 + 0.5, os.path.join(output_directory, f"night_{val_idx}.png"))
              
def parse_arguments():

    parser = argparse.ArgumentParser(description='Training Script')

    parser.add_argument('--generator_type', type=str, choices=['simple', 'sharing', 'unet', 'sharing_unet', 'pretrained_encoder'], default='simple',
                        help='Type of generator to use: simple, sharing, unet, sharing_unet or pretrained_encoder (default: simple)')
    parser.add_argument('--load_model', type=str, default='n',
                        help='Path to a checkpoint folder, e.g. "../outputs/training/checkpoints/epoch_200/" (default: n)')
    parser.add_argument('--loaded_epoch', type=str, default='n',
                        help='Number of epochs of training that have already been performed for the loaded model (default: n)')
    parser.add_argument('--disc_learning_rate', type=float, default=2e-4, 
                        help='Discriminator learning rate (default: 2e-4)')
    parser.add_argument('--gen_learning_rate', type=float, default=2e-4, 
                        help='Generator learning rate (default: 2e-4)')
    parser.add_argument('--lap_loss', type=str, choices=['y', 'n'], default='n',
                        help='Include a Laplacian Pyramid loss term: y or n')
    parser.add_argument('--mid_cycle_loss', type=str, choices=['y', 'n'], default='n',
                        help='Include a mid-cycle loss term to encourage a common latent representation (only applicable for shared encoder architecture): y or n')

    return parser.parse_args()

def main():
    
    args = parse_arguments()

    training_image_size = 256

    if args.generator_type == 'simple':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        gen_N = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
        gen_D = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
        summary(gen_N, (3, 256, 256), device=DEVICE)

    elif args.generator_type == 'sharing':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        encoder = enc.Encoder(img_channels=3, num_features=64).to(DEVICE)
        gen_N = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)
        gen_D = sh_gen.sharing_Generator(encoder, num_features=64, num_residuals=9, img_channels=3).to(DEVICE)
        summary(gen_N, (3, 256, 256), device=DEVICE)

    elif args.generator_type == 'unet':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        gen_N = un.UNet(input_channel=3, output_channel=3).to(DEVICE)
        gen_D = un.UNet(input_channel=3, output_channel=3).to(DEVICE)
        summary(gen_N, (3, 256, 256), device=DEVICE)

    elif args.generator_type == 'sharing_unet':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        #encoder = un_enc.UNet_Encoder(input_channel=3).to(DEVICE)
        #gen_N = un_dec.UNet_Decoder(encoder, output_channel=3).to(DEVICE)
        #gen_D = un_dec.UNet_Decoder(encoder, output_channel=3).to(DEVICE)
        #summary(encoder, (3, 256, 256), device=DEVICE)
        #summary(gen_N, (3, 256, 256), device=DEVICE)
        encoder = resn_enc.ResNet18Encoder().to(DEVICE)
        gen_N = resn_dec.ResNet18Decoder(encoder, output_channels=3).to(DEVICE)
        gen_D = resn_dec.ResNet18Decoder(encoder, output_channels=3).to(DEVICE)
        training_image_size = 224
        summary(encoder, (3, 224, 224), device=DEVICE)
        summary(gen_N, (3, 224, 224), device=DEVICE)

    elif args.generator_type == 'pretrained_encoder':
        disc_D = disc.Discriminator(in_channels=3).to(DEVICE)
        disc_N = disc.Discriminator(in_channels=3).to(DEVICE)
        gen_N = un_res.UnetResNet18(output_channels=3).to(DEVICE)
        gen_D = un_res.UnetResNet18(output_channels=3).to(DEVICE)
        training_image_size = 224 # keep size consistent with the training data of Resnet-18
        print(gen_N.base_model)
        summary (disc_N, (3, 224, 224), device=DEVICE)
        summary(gen_N, (3, 224, 224), device=DEVICE)

    # use Adam Optimizer for both generator and discriminator
    opt_disc = optim.Adam(
        list(disc_D.parameters()) + list(disc_N.parameters()), lr=args.disc_learning_rate, betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_N.parameters()) + list(gen_D.parameters()), lr=args.gen_learning_rate, betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    offset = 0 # no epoch offset if no loaded model (training from scratch), otherwise the loaded epoch must be added to the current epoch

    if args.load_model != 'n' and args.loaded_epoch == 'n':
        print("Invalid model loading arguments. You must specify the current epoch of the loaded checkpoint.")
        return

    if args.load_model != 'n':
        offset = int(args.loaded_epoch)

        gen_D_checkpoint = args.load_model + f"{CHECKPOINT_GENERATOR_D}_{args.loaded_epoch}.pth.tar"
        gen_N_checkpoint = args.load_model + f"{CHECKPOINT_GENERATOR_N}_{args.loaded_epoch}.pth.tar"
        disc_D_checkpoint = args.load_model + f"{CHECKPOINT_DISCRIMINATOR_D}_{args.loaded_epoch}.pth.tar"
        disc_N_checkpoint = args.load_model + f"{CHECKPOINT_DISCRIMINATOR_N}_{args.loaded_epoch}.pth.tar"

        load_checkpoint(gen_D_checkpoint, gen_D, opt_gen, args.gen_learning_rate)
        load_checkpoint(gen_N_checkpoint, gen_N, opt_gen, args.gen_learning_rate)
        load_checkpoint(disc_D_checkpoint, disc_D, opt_disc, args.disc_learning_rate)
        load_checkpoint(disc_N_checkpoint, disc_N, opt_disc, args.disc_learning_rate)
    
    day_training_transforms = ppd.set_training_transforms(training_image_size, day_means, day_stds)
    night_training_transforms = ppd.set_training_transforms(training_image_size, night_means, night_stds)
    day_val_transforms = ppd.set_val_transforms(day_means, day_stds)
    night_val_transforms = ppd.set_val_transforms(night_means, night_stds)

    dataset = ppd.DayNightDataset(
        root_day=TRAIN_DIR + "/day",
        root_night=TRAIN_DIR + "/night",
        size=training_image_size,
        #transform=transforms,
        day_transform=day_training_transforms,
        night_transform=night_training_transforms,
    )

    val_dataset = ppd.DayNightDataset(
        root_day=VAL_DIR + "/day",
        root_night=VAL_DIR + "/night",
        size=TEST_SIZE,
        #transform=ppd.val_transforms,
        day_transform=day_val_transforms,
        night_transform=night_val_transforms,
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

    laplace = False
    if args.lap_loss == 'y':
        laplace = True

    for epoch in range(NUM_EPOCHS):
        print('Epoch: ', (epoch+1), '/', NUM_EPOCHS)

        if args.mid_cycle_loss == 'n':
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
                laplace
            )
        
        elif args.mid_cycle_loss == 'y':
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
                laplace,
                encoder
            )
            
        if (epoch+1) % CHECKPOINT_INCREMENT == 0:
            epoch_folder = f"../outputs/training/checkpoints/epoch_{epoch+1+offset}/"

            if not os.path.exists(epoch_folder):
                os.makedirs(epoch_folder)

            # checkpoints start from epoch 0 unless a model was loaded, in which case the loaded epoch (E) is added to the name (epoch_10 becomes epoch_210 if we loaded epoch_200 checkpoints) 
            save_checkpoint(gen_D, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR_D}_{epoch+1+offset}.pth.tar"))
            save_checkpoint(gen_N, opt_gen, filename=os.path.join(epoch_folder, f"{CHECKPOINT_GENERATOR_N}_{epoch+1+offset}.pth.tar"))
            save_checkpoint(disc_D, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_D}_{epoch+1+offset}.pth.tar"))
            save_checkpoint(disc_N, opt_disc, filename=os.path.join(epoch_folder, f"{CHECKPOINT_DISCRIMINATOR_N}_{epoch+1+offset}.pth.tar"))

if __name__ == "__main__":
    seed_everything(100) # make the training as reproducible as possible
    main()
