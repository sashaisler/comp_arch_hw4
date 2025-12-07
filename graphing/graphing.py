import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.ticker import MultipleLocator

# -----------------------------
#  PLOT ONE SUBPLOT
# -----------------------------
def plot_radix(ax, items_thread, block_128, block_256, block_512,
               bit_size, y_min, y_max):
    # Lines
    ax.plot(items_thread, block_128, marker='o', markersize=8, linewidth=2,
            label="128 threads/block (4 warps)")
    ax.plot(items_thread, block_256, marker='o', markersize=8, linewidth=2,
            label="256 threads/block (8 warps)")
    ax.plot(items_thread, block_512, marker='o', markersize=8, linewidth=2,
            label="512 threads/block (16 warps)")

    # X axis
    ax.set_xlim(left=0, right=max(items_thread) + 2)  # small padding on right
    ax.set_xticks(items_thread)
    ax.tick_params(axis='x', labelsize=12)

    # Y axis: same limits for all subplots
    ax.set_ylim(y_min, y_max)

    # major ticks: every 200, with labels
    ax.yaxis.set_major_locator(MultipleLocator(200))
    # minor ticks: every 100, no labels
    ax.yaxis.set_minor_locator(MultipleLocator(100))

    ax.tick_params(axis='y', which='major', labelsize=12)
    # grid on both major and minor ticks
    ax.grid(True, which='both', linestyle='--', alpha=0.3)

    ax.set_xlabel("Items per Thread", fontsize=14)
    ax.set_ylabel("Runtime (ms)", fontsize=14)

    ax.set_title(f"{bit_size}-Bit Radix", fontsize=14)
    ax.legend(fontsize=10, loc='upper right')


# -----------------------------
#  DATA
# -----------------------------
items_thread = [2, 4, 8, 16, 32, 64]

block_128_8 = [832.67, 461.92, 278.75, 185.30, 145.59, 165.17]
block_256_8 = [476.42, 288.16, 195.35, 149.50, 138.18, 146.26]
block_512_8 = [357.89, 259.05, 214.16, 186.90, 177.90, 182.57]

block_128_4 = [1483.56, 641.45, 315.69, 227.33, 185.63, 162.78]
block_256_4 = [699.94, 374.09, 289.27, 245.12, 224.30, 219.77]
block_512_4 = [536.94, 447.21, 402.56, 384.33, 377.93, 384.77]

block_128_2 = [1629.01, 953.16, 609.98, 444.64, 360.46, 320.72]
block_256_2 = [1072.02, 729.35, 561.93, 479.29, 443.43, 433.02]
block_512_2 = [1053.84, 884.49, 802.02, 761.29, 753.43, 768.06]

# -----------------------------
#  GLOBAL Y-LIMITS
# -----------------------------
all_vals_global = (
    block_128_2 + block_256_2 + block_512_2 +
    block_128_4 + block_256_4 + block_512_4 +
    block_128_8 + block_256_8 + block_512_8
)

y_min = 0
max_val = max(all_vals_global)
# round up to next 200 and add a bit of headroom
y_max = 200 * int((max_val + 199) // 200)

# -----------------------------
#  MAKE ONE FIGURE WITH 3 SUBPLOTS
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

# left: 2-bit, middle: 4-bit, right: 8-bit
plot_radix(axes[0], items_thread, block_128_2, block_256_2, block_512_2, 2, y_min, y_max)
plot_radix(axes[1], items_thread, block_128_4, block_256_4, block_512_4, 4, y_min, y_max)
plot_radix(axes[2], items_thread, block_128_8, block_256_8, block_512_8, 8, y_min, y_max)

fig.tight_layout()

# Save single combined figure
folder = "figures"
if not os.path.exists(folder):
    os.makedirs(folder)
outpath = os.path.join(folder, "radix_all_bits.png")
fig.savefig(outpath, dpi=300)
print(f"Saved figure to: {outpath}")
# plt.show()  # uncomment to display interactively
