import tensorflow as tf
from tensorflow import keras
import sys
from preprocess_training_images import create_datasets

if len(sys.argv) != 6:
    print("Usage: python script.py <day_path> <night_path> <model> <start_epoch> <total_epochs>")
    sys.exit(1)
    
day_path = sys.argv[1]
night_path = sys.argv[2]

day_train_dataset, night_train_dataset = create_datasets(day_path, night_path)

if (sys.argv[3] == 'basic'):
    from basic import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_path = './model_checkpoints/basic/cyclegan_checkpoints.'
elif (sys.argv[3] == 'bilinear'):
    from bilinear import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_path = './model_checkpoints/bilinear/cyclegan_checkpoints.'
elif (sys.argv[3] == 'lp_loss'):
    from lp_loss import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_path = './model_checkpoints/lp_loss/cyclegan_checkpoints.'
elif (sys.argv[3] == 'benchmark'):
    from benchmark import CycleGan, gen_G, gen_F, disc_X, disc_Y, generator_loss_fn, discriminator_loss_fn
    checkpoint_path = './model_checkpoints/benchmark/cyclegan_checkpoints.'
else:
    print("Usage: invalid model name (try: 'bilinear')")
    sys.exit(1)

# Create cycle gan model
cycle_gan_model = CycleGan(
    generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y
)

# Compile the model
cycle_gan_model.compile(
    gen_G_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_F_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_X_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_Y_optimizer=keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),#
    gen_loss_fn=generator_loss_fn,
    disc_loss_fn=discriminator_loss_fn,
)

# Callbacks
checkpoint_filepath = checkpoint_path + "{epoch:03d}"
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True
)

# Define a function for loading the checkpoint and continuing training
def load_and_continue_training(model, checkpoint_path, initial_epoch, total_epochs):
    model.load_weights(checkpoint_path).expect_partial() # load the weights from the checkpoint
    print("Weights loaded successfully")
    # Continue training
    model.fit(
        tf.data.Dataset.zip((day_train_dataset, night_train_dataset)),
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[model_checkpoint_callback]
    )

# TRAINING FROM SCRATCH
if (int(sys.argv[4]) == 0):
    cycle_gan_model.fit(
        tf.data.Dataset.zip((day_train_dataset, night_train_dataset)),
        epochs=int(sys.argv[5]),
        callbacks=[model_checkpoint_callback]
    )
    
# TRAINING FROM CHECKPOINT
else:
    # Specify the checkpoint path, initial_epoch, and total_epochs
    last_epoch = int(sys.argv[4])
    start_checkpoint_path = checkpoint_path + f"{last_epoch:03d}"  # Adjust the checkpoint path
    total_epochs = int(sys.argv[5])  # Specify the total number of epochs you want to train
    
    # Call the function to load the checkpoint and continue training
    load_and_continue_training(cycle_gan_model, start_checkpoint_path, last_epoch, total_epochs)
