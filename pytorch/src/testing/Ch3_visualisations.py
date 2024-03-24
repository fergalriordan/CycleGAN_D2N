import matplotlib.pyplot as plt
import pandas as pd

# Load your dataframes
df1 = pd.read_csv('pytorch/src/testing/DnCNN.csv')
df2 = pd.read_csv('pytorch/src/testing/UNet.csv')
df3 = pd.read_csv('pytorch/src/testing/ResNet.csv')

# Set up the plotting
plt.figure(figsize=(8, 5))

plt.plot(df3['Epoch'], df3['FID_N_mean'], label='Night Images')
plt.plot(df3['Epoch'], df3['FID_D_mean'], label='Day Images')

# Shade the area between the two lines
plt.fill_between(df3['Epoch'], df3['FID_N_mean'], df3['FID_D_mean'], color='grey', alpha=0.3)

plt.ylim(100, 200)

plt.title('Mean FID values for ResNet-18 encoded generator')
plt.xlabel('Epochs')
plt.ylabel('FID')
plt.legend()
plt.grid(True)

# Show plot
plt.show()