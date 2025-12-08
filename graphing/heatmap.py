import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.colors import LogNorm


def make_heatmap(data_matrix, items_thread, block_labels, title, filename):
    # Ensure heatmaps folder exists
    folder = "heatmaps"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create figure
    plt.figure(figsize=(7, 3))

    ax = sns.heatmap(
        data_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn_r",
        cbar=True,
        norm=LogNorm(vmin=np.min(data_matrix), vmax=np.max(data_matrix)),
        # cbar_kws={"shrink": 0.8, "aspect": 30},
        linewidths=.5,
        annot_kws={"size": 12}  # number size inside tiles
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Runtime (ms)', fontsize=12, labelpad=15)

    # plt.title(title, fontsize=16)
    plt.xlabel("Threads per Block", fontsize=14, labelpad=15)
    plt.ylabel("Items per Thread", fontsize=14, labelpad=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


    # ---- Center X ticks ----
    num_cols = data_matrix.shape[1]
    x_tick_positions = [i + 0.5 for i in range(num_cols)]
    plt.xticks(x_tick_positions, block_labels, rotation=0)

    # ---- Center Y ticks ----
    num_rows = data_matrix.shape[0]
    y_tick_positions = [i + 0.5 for i in range(num_rows)]
    plt.yticks(y_tick_positions, items_thread, rotation=0)

    plt.tight_layout()

    # Save the figure
    outpath = os.path.join(folder, filename)
    plt.savefig(outpath, dpi=300)
    print(f"Saved heatmap to {outpath}")

    plt.close()  # prevent extra figures from appearing

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

data_8bit = np.array([
    block_128_8,
    block_256_8,
    block_512_8
]).T

data_4bit = np.array([
    block_128_4,
    block_256_4,
    block_512_4
]).T

data_2bit = np.array([
    block_128_2,
    block_256_2,
    block_512_2
]).T

make_heatmap(
    data_matrix=data_8bit,
    items_thread=items_thread,
    block_labels=["128 (4 warps)", "256 (8 warps)", "512 (16 warps)"],
    title="8-Bit Radix Sort Runtime Heatmap (in ms)",
    filename="radix_8bit_heatmap.png"
)

make_heatmap(
    data_matrix=data_4bit,
    items_thread=items_thread,
    block_labels=["128 (4 warps)", "256 (8 warps)", "512 (16 warps)"],
    title="4-Bit Radix Sort Runtime Heatmap (in ms)",
    filename="radix_4bit_heatmap.png"
)

make_heatmap(
    data_matrix=data_2bit,
    items_thread=items_thread,
    block_labels=["128 (4 warps)", "256 (8 warps)", "512 (16 warps)"],
    title="2-Bit Radix Sort Runtime Heatmap (in ms)",
    filename="radix_2bit_heatmap.png"
)