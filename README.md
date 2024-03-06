# CycleGAN_D2N
Repository for the code involved in the 30 ECTS Research Project completed as part my of MAI in Electronic and Computer Engineering. The project involves the investigation of the applicability of cycle-consistent generative adversarial networks (CycleGAN) for image-to-image translation between domains with significant domain gaps, such as daytime and nighttime. The project also involves the investigation of architectural and training process adaptations that can be made to a basic CycleGAN model for improved performance in distant-domain translation, such as a shared encoder and a mid-cycle consistency loss. 

Original experimentation was performed with a TensorFlow implementation (can be found in the archive folder) but the research discussed in my MAI thesis was performed using the PyTorch code found in the pytorch directory. 

# Datasets

The primary dataset used in this project is the "Unpaired Day and Night Cityview Images" dataset found at https://www.kaggle.com/datasets/heonh0/daynight-cityview. 
This dataset was augmented with some additional night images from the "Aachen Day-Night" dataset found at https://paperswithcode.com/dataset/aachen-day-night. 

# Training

Change working directory to pytorch/src, then run the command "python training/train.py", using command line arguments to specify the desired hyperparameters. 

# Testing

To generate high-resolution validation images, change working directory to pytorch/src, then run the command "python testing/generate_images.py --model_type --model_path --epoch". This script will load a pretrained model of the given type and output high-resolution images to the outputs/testing directory. 

To compare model performances, quantitative metrics (Kernel Inception Distance, Fréchet Inception Distance) are calculated using the src/testing/metrics.py script. This is still a work in progress. 