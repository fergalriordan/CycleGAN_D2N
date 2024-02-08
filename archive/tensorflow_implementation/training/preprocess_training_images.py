# preprocess_training_images.py

import tensorflow as tf
 
orig_img_size = (512, 512)  # standard input image size
input_img_size = (400, 400, 3)  # size of random crops to use for training

def normalize_img(img):
        img = tf.cast(img, dtype=tf.float32)
        return (img / 127.5) - 1.0 # map values in the range [-1, 1]
        
# Define preprocessing functions
def preprocess_train_image(img, label):
    img = tf.image.random_flip_left_right(img) # random flip
    img = tf.image.resize(img, [orig_img_size[0], orig_img_size[1]]) # resize all images to the standard size
    img = tf.image.random_crop(img, size=[input_img_size[0], input_img_size[1], input_img_size[2]]) # random crop for training
    img = normalize_img(img) # normalize the pixel values [-1, 1]
    return img

def create_datasets(day_path, night_path):

    # Load the images.
    day_images = tf.data.Dataset.list_files(day_path + '/*.jpg')
    night_images = tf.data.Dataset.list_files(night_path + '/*.jpg')

    # Define buffer size and batch size
    buffer_size = 32
    batch_size = 1
    # Autotune for parallel processing
    autotune = tf.data.AUTOTUNE
    
    day_train_dataset = (day_images
        .shuffle(buffer_size)
        .take(len(list(day_images)))
        .map(lambda x: (tf.image.decode_jpeg(tf.io.read_file(x)), 0), num_parallel_calls=autotune)
        .map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .batch(batch_size)
    )
    
    night_train_dataset = (night_images
        .shuffle(buffer_size)
        .take(len(list(night_images)))
        .map(lambda x: (tf.image.decode_jpeg(tf.io.read_file(x)), 1), num_parallel_calls=autotune)
        .map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .batch(batch_size)
    )

    return day_train_dataset, night_train_dataset

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python preprocess_training_images.py <day_path> <night_path>")
        sys.exit(1)

    day_path = sys.argv[1]
    night_path = sys.argv[2]

    # Create and return the datasets
    day_train_dataset, night_train_dataset = create_datasets(day_path, night_path)
