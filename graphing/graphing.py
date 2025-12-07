import matplotlib.pyplot as plt
import os

# -----------------------------
#  SAVE FIGURE TO FOLDER
# -----------------------------
def save_plot(bit_size, folder="figures"):
    filename="radix_" + str(bit_size) + "bit.png"
    # Create folder if not exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Full file path
    path = os.path.join(folder, filename)
    
    # Save the figure
    plt.savefig(path, dpi=300)  # 300 DPI = publication quality
    print(f"Saved figure to: {path}")

# -----------------------------
#  PLOT
# -----------------------------
def plot_radix(items_thread, block_128, block_256, block_512, bit_size):
    plt.figure(figsize=(8, 4.5))
    plt.plot(items_thread, block_128, marker='o', markersize=8, linewidth=2,
             label="128 threads/block (4 warps)")
    plt.plot(items_thread, block_256, marker='o', markersize=8, linewidth=2,
             label="256 threads/block (8 warps)")
    plt.plot(items_thread, block_512, marker='o', markersize=8, linewidth=2,
             label="512 threads/block (16 warps)")

    # Fix x-axis alignment
    plt.xlim(left=0)

    plt.xticks(items_thread, fontsize=14)  # tick marks at exactly 1,2,3,4,5,6
    plt.yticks(fontsize=14)

    plt.xlabel("Items per Thread", fontsize=16, labelpad=10)
    plt.ylabel("Runtime (ms)", fontsize=16, labelpad=10)
    # plt.title(str(bit_size) + "-Bit Radix Sort Performance", fontsize=18)

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=14, loc='upper right')
    plt.tight_layout()

    # Call the save function
    save_plot(bit_size)

    # Show the plot
    # plt.show()


# -----------------------------
#  YOUR 8-BIT DATA
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


plot_radix(items_thread, block_128_8, block_256_8, block_512_8, 8)
plot_radix(items_thread, block_128_4, block_256_4, block_512_4, 4)
plot_radix(items_thread, block_128_2, block_256_2, block_512_2, 2)