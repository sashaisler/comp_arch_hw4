#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ======================================================================
// Kernel A: per-block histogram in shared memory
// ======================================================================

__global__
void buildBlockHistKernel(const uint32_t *d_in,
                          int n,
                          int shift,
                          unsigned int *d_blockHist,
                          size_t radix,
                          uint32_t radixMask,
                          int chunkSize)
{
    extern __shared__ unsigned int s_hist[];

    int tid        = threadIdx.x;
    int blockId    = blockIdx.x;
    int blockStart = blockId * chunkSize;
    int blockEnd   = blockStart + chunkSize;
    if (blockStart >= n) return;
    if (blockEnd > n) blockEnd = n;

    // Zero per-block histogram
    for (size_t b = tid; b < radix; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // Build local histogram
    for (int idx = blockStart + tid; idx < blockEnd; idx += blockDim.x) {
        uint32_t v       = d_in[idx];
        unsigned int buk = (v >> shift) & radixMask;
        atomicAdd(&s_hist[buk], 1);
    }
    __syncthreads();

    // Write to global memory: d_blockHist[blockId * radix + b]
    size_t base = (size_t)blockId * radix;
    for (size_t b = tid; b < radix; b += blockDim.x) {
        d_blockHist[base + b] = s_hist[b];
    }
}

// ======================================================================
// Kernel: reduce per-block histograms into global bucketTotals[b]
// ======================================================================

__global__
void reduceBucketsKernel(const unsigned int *d_blockHist,
                         unsigned int *d_bucketTotals,
                         int numBlocks,
                         size_t radix)
{
    size_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= radix) return;

    unsigned int sum = 0;
    for (int blk = 0; blk < numBlocks; ++blk) {
        sum += d_blockHist[(size_t)blk * radix + bucket];
    }
    d_bucketTotals[bucket] = sum;
}

// ======================================================================
// Kernel: exclusive scan of bucketTotals -> bucketBase
// Assumes radix is a power of two and <= maxThreadsPerBlock.
// ======================================================================

__global__
void exclusiveScanKernel(const unsigned int *in,
                         unsigned int *out,
                         size_t n)
{
    extern __shared__ unsigned int temp[];

    int thid = threadIdx.x;
    if ((size_t)thid >= n) return;

    temp[thid] = in[thid];

    int offset = 1;

    // upsweep
    for (size_t d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if ((size_t)thid < d) {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }

    if (thid == 0) {
        temp[n - 1] = 0;
    }

    // downsweep
    for (size_t d = 1; d < n; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if ((size_t)thid < d) {
            int ai = offset * ((thid << 1) + 1) - 1;
            int bi = offset * ((thid << 1) + 2) - 1;
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    out[thid] = temp[thid];
}

// ======================================================================
// Kernel: compute per-block offsets from block hist + bucketBase
// ======================================================================

__global__
void computeBlockOffsetsKernel(const unsigned int *d_blockHist,
                               const unsigned int *d_bucketBase,
                               unsigned int *d_blockOffsets,
                               int numBlocks,
                               size_t radix)
{
    size_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
    if (bucket >= radix) return;

    unsigned int running = d_bucketBase[bucket];

    for (int blk = 0; blk < numBlocks; ++blk) {
        size_t idx = (size_t)blk * radix + bucket;
        unsigned int h = d_blockHist[idx];
        d_blockOffsets[idx] = running;
        running += h;
    }
}

// ======================================================================
// Kernel B: scatter using per-block offsets
// ======================================================================

__global__
void scatterKernel(const uint32_t *d_in,
                   uint32_t *d_out,
                   int n,
                   int shift,
                   const unsigned int *d_blockOffsets,
                   size_t radix,
                   uint32_t radixMask,
                   int chunkSize)
{
    extern __shared__ unsigned int localCount[];

    int tid        = threadIdx.x;
    int blockId    = blockIdx.x;
    int blockStart = blockId * chunkSize;
    int blockEnd   = blockStart + chunkSize;
    if (blockStart >= n) return;
    if (blockEnd > n) blockEnd = n;

    // Zero per-block local counters
    for (size_t b = tid; b < radix; b += blockDim.x) {
        localCount[b] = 0;
    }
    __syncthreads();

    const unsigned int *blockOffsetBase = d_blockOffsets + (size_t)blockId * radix;

    // Stable scatter within this block
    if (tid == 0) {
        for (int idx = blockStart; idx < blockEnd; ++idx) {
            uint32_t v       = d_in[idx];
            unsigned int buk = (v >> shift) & radixMask;

            unsigned int pos = blockOffsetBase[buk] + localCount[buk];
            localCount[buk]++;

            d_out[pos] = v;
        }
    }
}

// ======================================================================
// Fully GPU-based LSD radix sort
// ======================================================================

void radixSortGPU(uint32_t *d_data,
                  int n,
                  int radixBits,
                  int blockSize,
                  int itemsPerThread,
                  float *elapsed_ms_out)
{
    if (radixBits <= 0 || radixBits > 32 || (32 % radixBits) != 0) {
        fprintf(stderr,
            "Error: radixBits must be > 0, <= 32, and divide 32 (got %d)\n",
            radixBits);
        exit(EXIT_FAILURE);
    }

    if (blockSize <= 0 || blockSize > 1024 || (blockSize % 32) != 0) {
        fprintf(stderr,
            "Error: blockSize must be > 0, <= 1024, and a multiple of 32 (got %d)\n",
            blockSize);
        exit(EXIT_FAILURE);
    }

    if (itemsPerThread <= 0) {
        fprintf(stderr,
            "Error: itemsPerThread must be > 0 (got %d)\n",
            itemsPerThread);
        exit(EXIT_FAILURE);
    }

    int chunkSize = blockSize * itemsPerThread;
    if (chunkSize <= 0) {
        fprintf(stderr, "Error: chunkSize overflow or invalid\n");
        exit(EXIT_FAILURE);
    }

    size_t radix = (size_t)1 << radixBits;

    int maxThreadsPerBlock = 1024;
    if (radix > (size_t)maxThreadsPerBlock) {
        fprintf(stderr,
            "Error: radix (%zu) too large for simple single-block scan kernel.\n"
            "Use smaller radixBits or implement a multi-block scan.\n",
            radix);
        exit(EXIT_FAILURE);
    }

    uint32_t radixMask;
    if (radixBits == 32) {
        radixMask = 0xFFFFFFFFu;
    } else {
        radixMask = (uint32_t)(((uint64_t)1 << radixBits) - 1u);
    }

    int numBlocks = (n + chunkSize - 1) / chunkSize;

    uint32_t *d_temp = NULL;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    unsigned int *d_blockHist    = NULL;
    unsigned int *d_blockOffsets = NULL;
    unsigned int *d_bucketTotals = NULL;
    unsigned int *d_bucketBase   = NULL;

    CUDA_CHECK(cudaMalloc(&d_blockHist,    (size_t)numBlocks * radix * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, (size_t)numBlocks * radix * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bucketTotals, radix * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bucketBase,   radix * sizeof(unsigned int)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    int numPasses = 32 / radixBits;
    uint32_t *d_in  = d_data;
    uint32_t *d_out = d_temp;

    for (int pass = 0; pass < numPasses; ++pass) {
        int shift = pass * radixBits;

        size_t shmem_hist = radix * sizeof(unsigned int);
        buildBlockHistKernel<<<numBlocks, blockSize, shmem_hist>>>(
            d_in, n, shift, d_blockHist, radix, radixMask, chunkSize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        int threadsBuckets = 256;
        int blocksBuckets  = (int)((radix + threadsBuckets - 1) / threadsBuckets);
        reduceBucketsKernel<<<blocksBuckets, threadsBuckets>>>(
            d_blockHist, d_bucketTotals, numBlocks, radix
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        exclusiveScanKernel<<<1, (int)radix, radix * sizeof(unsigned int)>>>(
            d_bucketTotals, d_bucketBase, radix
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        computeBlockOffsetsKernel<<<blocksBuckets, threadsBuckets>>>(
            d_blockHist, d_bucketBase, d_blockOffsets, numBlocks, radix
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        size_t shmem_scatter = radix * sizeof(unsigned int);
        scatterKernel<<<numBlocks, blockSize, shmem_scatter>>>(
            d_in, d_out, n, shift, d_blockOffsets, radix, radixMask, chunkSize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t *tmp = d_in;
        d_in  = d_out;
        d_out = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms_out) *elapsed_ms_out = elapsed_ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    if (d_in != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, d_in, n * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_blockOffsets));
    CUDA_CHECK(cudaFree(d_bucketTotals));
    CUDA_CHECK(cudaFree(d_bucketBase));
}

// ======================================================================
// main
// Usage: ./prog <radix_bits> <block_size> <items_per_thread> <input_file>
// ======================================================================

int main(int argc, char **argv)
{
    const int N = 1 << 29; // 268,435,456 elements

    if (argc != 5) {
        fprintf(stderr,
                "Usage: %s <radix_bits> <block_size> <items_per_thread> <input_file>\n",
                argv[0]);
        return 1;
    }

    int radixBits      = atoi(argv[1]);
    int blockSize      = atoi(argv[2]);
    int itemsPerThread = atoi(argv[3]);
    const char *inputPath = argv[4];

    if (radixBits <= 0 || radixBits > 32 || (32 % radixBits) != 0) {
        fprintf(stderr,
                "Error: radix_bits must be > 0, <= 32, and divide 32 (got %d)\n",
                radixBits);
        return 1;
    }

    if (blockSize <= 0 || blockSize > 1024 || (blockSize % 32) != 0) {
        fprintf(stderr,
                "Error: block_size must be > 0, <= 1024, and a multiple of 32 (got %d)\n",
                blockSize);
        return 1;
    }

    if (itemsPerThread <= 0) {
        fprintf(stderr,
                "Error: items_per_thread must be > 0 (got %d)\n",
                itemsPerThread);
        return 1;
    }

    uint32_t *h_in  = (uint32_t*)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    FILE *fin = fopen(inputPath, "r");
    if (!fin) {
        perror("Error opening input file");
        free(h_in);
        free(h_out);
        return 1;
    }

    for (int i = 0; i < N; ++i) {
        if (fscanf(fin, "%u", &h_in[i]) != 1) {
            fprintf(stderr,
                    "Error: input file '%s' contains fewer than %d integers (stopped at %d)\n",
                    inputPath, N, i);
            fclose(fin);
            free(h_in);
            free(h_out);
            return 1;
        }
    }
    fclose(fin);

    uint32_t *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in, N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    float elapsed_ms = 0.0f;
    radixSortGPU(d_data, N, radixBits, blockSize, itemsPerThread, &elapsed_ms);

    printf("GPU radix sort time (%d-bit passes, blockSize=%d, itemsPerThread=%d): %.3f ms\n",
           radixBits, blockSize, itemsPerThread, elapsed_ms);

    CUDA_CHECK(cudaMemcpy(h_out, d_data, N * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    int fail_idx = -1;
    for (int i = 1; i < N; ++i) {
        if (h_out[i - 1] > h_out[i]) {
            fail_idx = i;
            break;
        }
    }

    if (fail_idx == -1) {
        printf("Sort verification PASSED\n");
    } else {
        printf("Sort verification FAILED at index %d (values %u, %u)\n",
               fail_idx, h_out[fail_idx - 1], h_out[fail_idx]);
    }

    CUDA_CHECK(cudaFree(d_data));
    free(h_in);
    free(h_out);

    return 0;
}
