import matplotlib.pyplot as plt
import pandas as pd

# Load your dataframes
df1 = pd.read_csv('pytorch/src/testing/Ch3_metric_data/DnCNN.csv')
df2 = pd.read_csv('pytorch/src/testing/Ch3_metric_data/UNet.csv')
df3 = pd.read_csv('pytorch/src/testing/Ch3_metric_data/ResNet.csv')

# Set up the plotting
plt.figure(figsize=(8, 5))

plt.plot(df1['Epoch'], df1['FID_N_mean'], label='Night Images')
plt.plot(df1['Epoch'], df1['FID_D_mean'], label='Day Images')

# Shade the area between the two lines
plt.fill_between(df1['Epoch'], df1['FID_N_mean'], df1['FID_D_mean'], color='grey', alpha=0.3)

plt.ylim(100, 200)

plt.title('Mean FID values for Original CycleGAN generator')
plt.xlabel('Epochs')
plt.ylabel('FID')
plt.legend()
plt.grid(True)

# Show plot
plt.show()