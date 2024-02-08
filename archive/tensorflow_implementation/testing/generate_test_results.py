import tensorflow as tf
import sys
import os
from PIL import Image
import numpy as np

if len(sys.argv) != 3:
    print('Usage: python generate_test_results.py <test_images_path> <model>')
    sys.exit(1)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__ or sys.argv[0]))

# Go up one level to the project root and then import from the models directory
sys.path.append(os.path.join(script_dir, '..'))

test_path = sys.argv[1]

if sys.argv[2] == 'basic':
    from models.basic import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_dir = os.path.join(script_dir, '..', 'model_checkpoints', 'basic')
    save_path = os.path.join('test_data', 'results', 'benchmark_ablation_study', 'basic')
elif sys.argv[2] == 'bilinear':
    from models.bilinear import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_dir = os.path.join(script_dir, '..', 'model_checkpoints', 'bilinear')
    save_path = os.path.join('test_data', 'results', 'benchmark_ablation_study', 'bilinear')
elif sys.argv[2] == 'lp_loss':
    from models.lp_loss import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_dir = os.path.join(script_dir, '..', 'model_checkpoints', 'lp_loss')
    save_path = os.path.join('test_data', 'results', 'benchmark_ablation_study', 'lp_loss')
elif sys.argv[2] == 'benchmark':
    from models.benchmark import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_dir = os.path.join(script_dir, '..', 'model_checkpoints', 'benchmark')
    save_path = os.path.join('test_data', 'results', 'benchmark_ablation_study', 'benchmark')
else:
    print("Usage: invalid model name (try: 'bilinear')")
    sys.exit(1)

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# load test images and generate test results for multiple epochs
batch_size = 1
autotune = tf.data.AUTOTUNE

def normalize_img(img):
        img = tf.cast(img, dtype=tf.float32)
        return (img / 127.5) - 1.0 # map values in the range [-1, 1]

# define functions to test and save results
def preprocess_test_image(img, label):
    img = normalize_img(img) # normalize the pixel values [-1, 1]
    return img

def save_image(img, output_dir, filename):
        im = Image.fromarray(img)
        im.save(os.path.join(output_dir, filename))

def test(test_dataset, path):
  os.makedirs(path, exist_ok=True)
  for i, img in enumerate(test_dataset):
    prediction = cycle_gan_model.gen_G(img, training=False)[0].numpy()
    prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
    save_image(prediction, path, f"{i}.png")

for epoch in range(10, 201, 10):
    checkpoint_path = os.path.join(checkpoint_dir, f'cyclegan_checkpoints.{epoch:03d}')
    save_epoch_path = os.path.join(save_path, f'epoch{epoch:03d}')

    # Load the checkpoints
    cycle_gan_model.load_weights(checkpoint_path).expect_partial()
    print(f"Weights for epoch {epoch} loaded successfully")

    # Load test images and preprocess them
    test_images = tf.data.Dataset.list_files(test_path + '/*.jpg')
    test_dataset = (test_images
        .take(len(test_images))
        .map(lambda x: (tf.image.decode_jpeg(tf.io.read_file(x)), 0), num_parallel_calls=autotune)
        .map(preprocess_test_image, num_parallel_calls=autotune)
        .cache()
        .batch(batch_size)
    )

    # Generate and save test results
    test(test_dataset, save_epoch_path)
