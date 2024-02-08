from PIL import Image
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from models import generator as gen 
from models import preprocess_data as ppd  

# Set the device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Paths for test data
TEST_DIR = "../data/val"  # Replace with the path to your test data

# Path for pre-trained model checkpoints
GEN_NIGHT_CHECKPOINT = "../outputs/training/checkpoints/epoch_200/gen_n_200.pth.tar"  # Replace with the path to your generator checkpoint
GEN_DAY_CHECKPOINT = "../outputs/training/checkpoints/epoch_200/gen_d_200.pth.tar"  # Replace with the path to your generator checkpoint

# Load pre-trained generator models
gen_Night = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)
gen_Day = gen.Generator(img_channels=3, num_residuals=9).to(DEVICE)

# Load the pre-trained weights
checkpoint_Night = torch.load(GEN_NIGHT_CHECKPOINT, map_location=DEVICE)
checkpoint_Day = torch.load(GEN_DAY_CHECKPOINT, map_location=DEVICE)

gen_Night.load_state_dict(checkpoint_Night["state_dict"])
gen_Day.load_state_dict(checkpoint_Day["state_dict"])

# Set models to evaluation mode
gen_Night.eval()
gen_Day.eval()


test_image_transforms = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False,  # if set to True, the width and height need to be the same (square image)
)


# Create a test dataset
test_dataset = ppd.DayNightDataset(
    root_day=TEST_DIR + "/day",
    root_night=TEST_DIR + "/night",
    size=None,
    transform=test_image_transforms,
)

# Create a test data loader
test_loader = DataLoader(
   test_dataset, 
   batch_size=1, 
   shuffle=False, 
   pin_memory=True,
)

# Test the model on the test dataset
for idx, (night_img, day_img) in enumerate(test_loader):
    night_img = night_img.to(DEVICE)
    day_img = day_img.to(DEVICE)

    # Generate images using generators
    with torch.no_grad():
      generated_day_img = gen_Day(night_img)
      generated_night_img = gen_Night(day_img)

    # Save the generated images if needed
    save_image(generated_day_img * 0.5 + 0.5, f"..outputs/testing/day_{idx}.png")
    save_image(generated_night_img * 0.5 + 0.5, f"..outputs/testing/night_{idx}.png")

    torch.cuda.empty_cache()
