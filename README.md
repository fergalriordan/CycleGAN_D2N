# CycleGAN_D2N
Repository for the code involved in the 30 ECTS Research Project completed as part my of MAI in Electronic and Computer Engineering. The project involves the investigation of the applicability of cycle-consistent generative adversarial networks (CycleGAN) for image-to-image translation between domains with significant domain gaps, such as daytime and night-time. The project also involves the investigation of architectural and training process adaptations that can be made to a basic CycleGAN model for improved performance in distant-domain translation, such as a shared encoder and a mid-cycle consistency loss. 

Original experimentation was performed with a TensorFlow implementation (can be found in the archive folder) but the research discussed in my MAI thesis was performed using the PyTorch code found in the pytorch directory. 

# Datasets

The primary dataset used in this project is the "Unpaired Day and Night Cityview Images" dataset found at https://www.kaggle.com/datasets/heonh0/daynight-cityview. 
This dataset was augmented with some additional night images from the "Aachen Day-Night" dataset found at https://paperswithcode.com/dataset/aachen-day-night. 

# Preprocessing

The script for preprocessing the training data is located in the pytorch/src/preprocessing directory. It applies a series of pre-processing steps to augment the training set. 

# Models

The models that were implemented in this project can be found in the pytorch/src/models directory. The models are as follows: 

** discriminator.py - The adversarial PatchGAN discriminator
** generator.py - The original CycleGAN generator
** unet.py - A basic U-Net generator
** unet_encoder.py, unet_decoder.py - U-Net encoder and decoder scripts that were used to experiment with encoder sharing. Note that these scripts were not used to generate any results in the final report: the ResNet-18 encoder was shared instead. 
** unet_resnet_encoder.py - A U-Net generator with a pre-trained ResNet-18 encoder
** resnet_encoder.py, resnet_decoder.py - ResNet-18 encoder and a corresponding decoder script for use in the encoder-sharing experiments. 
** timestamped_unet_decoder - Decoder for a timestamped generator that doesn't use a ResNet-18 encoder. Again, this was not used to generate any results in the final report, it was merely used in an experimental fashion in the earlier stages of the project
** timestamped_resnet_decoder - Decoder for a timestamped generator with a ResNet-18 encoder. 

# Training

Change working directory to pytorch/src, then run the command "python training/train.py", using command line arguments to specify the desired hyperparameters. 

Note that a script for a Laplacian pyramid loss term is also present in the training directory. Some experimentation was performed with this loss term but ultimately the results were not discussed in the final report. 

# Testing

To generate high-resolution validation images, change working directory to pytorch/src, then run the command "python testing/generate_images.py --model_type --model_path --epoch". This script will load a pretrained model of the given type and output high-resolution images to the outputs/testing directory. 

To compare model performances, quantitative metrics (Kernel Inception Distance, Fréchet Inception Distance) are calculated using the src/testing/metrics.py script. Other operations such as generatign plots of the metric data can also be performed using the other scripts in the testing folder. 

# Outputs and Training Data Storage

The outputs from the training and testing scripts are stored in the pytorch/outputs directory. The training data is stored in the pytorch/data folder. 

# References

** The Keras example that was used in the early, exploratory stages: https://keras.io/examples/generative/cyclegan/ 

** The PyTorch implementation of CycleGAN that ultimately formed the foundation of this project: https://medium.com/@chilldenaya/cyclegan-introduction-pytorch-implementation-5b53913741ca

** The U-Net generator architecture was adapted from: https://github.com/a7med12345/Cycle-GAN-with-Unet-as-GENERATOR


** The script for computing a Laplacian Pyramid loss term was taken from: https://gist.github.com/alper111/b9c6d80e2dba1ee0bfac15eb7dad09c8?permalink_comment_id=4619133 