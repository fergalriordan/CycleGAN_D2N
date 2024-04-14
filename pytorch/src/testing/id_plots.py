import matplotlib.pyplot as plt
import numpy as np

# Identity loss weights
mid_weights = np.array([0, 1, 2, 3])

# KID Mean (D2N and N2D)
kid_mean_d2n = np.array([0.0719, 0.0674, 0.0476, 0.0828])
kid_mean_n2d = np.array([0.0808, 0.0815, 0.0852, 0.1120])

# FID Mean (D2N and N2D)
fid_mean_d2n = np.array([128.01, 126.35, 108.68, 131.46])
fid_mean_n2d = np.array([134.86, 139.66, 138.92, 162.49])

# Plotting KID Metrics
plt.figure(figsize=(14, 6))

# Plotting FID Metrics
plt.subplot(1, 2, 1)
plt.plot(mid_weights, fid_mean_d2n, label='Night Images', color='blue', linestyle='-', marker='o')
plt.plot(mid_weights, fid_mean_n2d, label='Day Images', color='blue', linestyle='--', marker='o')
plt.title('Fréchet Inception Distance (FID) vs. Mid-Cycle Loss Weight')
plt.xlabel('Mid-Cycle Loss Weight', fontsize=12)
plt.ylabel('Mean FID', fontsize=12)
plt.ylim(100, 200)
plt.legend()
plt.grid(True)


plt.subplot(1, 2, 2)
plt.errorbar(mid_weights, kid_mean_d2n, label='Night Images', color='blue', linestyle='-', marker='o')
plt.errorbar(mid_weights, kid_mean_n2d, label='Day Images', color='blue', linestyle='--', marker='o')
plt.title('Kernel Inception Distance (KID) vs. Mid-Cycle Loss Weight')
plt.xlabel('Mid-Cycle Loss Weight', fontsize=12)
plt.ylabel('Mean KID', fontsize=12)
plt.ylim(0.02, 0.16)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
