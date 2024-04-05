import matplotlib.pyplot as plt
import numpy as np

# Identity loss weights
id_weights = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

# KID Mean (D2N and N2D)
kid_mean_d2n = np.array([0.0688, 0.0612, 0.0633, 0.0587, 0.0549, 0.0652, 0.0588, 0.0881, 0.0847, 0.0567])
kid_mean_n2d = np.array([0.0848, 0.0940, 0.1020, 0.0991, 0.0875, 0.1036, 0.0987, 0.0845, 0.1034, 0.1147])

# FID Mean (D2N and N2D)
fid_mean_d2n = np.array([128.13, 120.73, 116.68, 114.94, 113.66, 119.52, 118.35, 138.15, 141.15, 109.91])
fid_mean_n2d = np.array([138.64, 145.234, 151.79, 147.77, 139.82, 153.10, 149.16, 140.51, 150.47, 162.69])

# Plotting KID Metrics
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.errorbar(id_weights, kid_mean_d2n, label='Night Images')
plt.errorbar(id_weights, kid_mean_n2d, label='Day Images')
plt.title('Kernel Inception Distance (KID)')
plt.xlabel('Identity Loss Weight')
plt.ylabel('Mean KID')
plt.fill_between(id_weights, kid_mean_d2n, kid_mean_n2d, color='grey', alpha=0.3)
plt.ylim(0.02, 0.16)
plt.legend()
plt.grid(True)

# Plotting FID Metrics
plt.subplot(1, 2, 2)
plt.plot(id_weights, fid_mean_d2n, label='Night Images')
plt.plot(id_weights, fid_mean_n2d, label='Day Images')
plt.title('Fréchet Inception Distance (FID)')
plt.xlabel('Identity Loss Weight')
plt.ylabel('Mean FID')
plt.fill_between(id_weights, fid_mean_d2n, fid_mean_n2d, color='grey', alpha=0.3)
plt.ylim(100, 200)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
