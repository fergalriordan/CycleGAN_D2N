import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('pytorch/outputs/testing/Ch4_metric_data/ResNet.csv')

plt.figure(figsize=(8, 5))

plt.plot(df['Epoch'], df['FID_N_mean'], label='Night Images', linestyle='-', color='blue')
plt.plot(df['Epoch'], df['FID_D_mean'], label='Day Images', linestyle='--', color='blue')

# Shade the area between the two lines
#plt.fill_between(df1['Epoch'], df1['FID_N_mean'], df1['FID_D_mean'], color='grey', alpha=0.3)

plt.ylim(100, 200)

plt.title('Fréchet Inception Distance (FID) vs. Epochs')
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('FID', fontsize=12)
plt.legend()
plt.grid(True)

# Show plot
plt.show()