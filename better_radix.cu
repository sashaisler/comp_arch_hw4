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

#define BLOCK_SIZE 256          // threads per block
#define ITEMS_PER_THREAD 4      // how many items each thread processes
#define CHUNK_SIZE (BLOCK_SIZE * ITEMS_PER_THREAD)

// ================= Kernel A: per-block histogram =================
// Each block processes CHUNK_SIZE contiguous elements (except maybe last).
// All threads in the block cooperatively build a shared histogram.

__global__
void buildBlockHistKernel(const uint32_t *d_in,
                          int n,
                          int shift,
                          unsigned int *d_blockHist,
                          size_t radix,
                          uint32_t radixMask)
{
    extern __shared__ unsigned int s_hist[];

    int tid        = threadIdx.x;
    int blockId    = blockIdx.x;
    int blockStart = blockId * CHUNK_SIZE;
    int blockEnd   = blockStart + CHUNK_SIZE;
    if (blockEnd > n) blockEnd = n;

    // Initialize shared histogram
    for (size_t b = tid; b < radix; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    // Accumulate local histogram
    for (int idx = blockStart + tid; idx < blockEnd; idx += blockDim.x) {
        uint32_t v       = d_in[idx];
        unsigned int buk = (v >> shift) & radixMask;
        atomicAdd(&s_hist[buk], 1);
    }
    __syncthreads();

    // Write shared histogram to global memory
    size_t base = (size_t)blockId * radix;
    for (size_t b = tid; b < radix; b += blockDim.x) {
        d_blockHist[base + b] = s_hist[b];
    }
}

// ================= Kernel B: parallel scatter per block =================
// Each block scatters its CHUNK_SIZE elements into the correct positions.
// We use per-block per-bucket base offsets (precomputed on host) plus
// per-block shared counters so that *all* threads in the block can
// scatter in parallel.
// NOTE: This is NOT stable within a bucket anymore, but final numeric
// order is correct (stability doesn't matter for plain integers).

__global__
void scatterKernel(const uint32_t *d_in,
                   uint32_t *d_out,
                   int n,
                   int shift,
                   const unsigned int *d_blockOffsets,
                   size_t radix,
                   uint32_t radixMask)
{
    extern __shared__ unsigned int localCount[];

    int tid        = threadIdx.x;
    int blockId    = blockIdx.x;
    int blockStart = blockId * CHUNK_SIZE;
    int blockEnd   = blockStart + CHUNK_SIZE;
    if (blockEnd > n) blockEnd = n;

    // Initialize per-block bucket counters cooperatively
    for (size_t b = tid; b < radix; b += blockDim.x) {
        localCount[b] = 0;
    }
    __syncthreads();

    const unsigned int *blockOffsetBase = d_blockOffsets + (size_t)blockId * radix;

    // To keep the pass STABLE, we must process elements in block order.
    // Let a single thread (tid == 0) walk [blockStart, blockEnd) in order.
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

// ================= Generic LSD radix sort =================

void radixSortGPU(uint32_t *d_data, int n, int radixBits, float *elapsed_ms_out)
{
    // Temporary array (ping-pong)
    uint32_t *d_temp = NULL;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    // Number of blocks: each block handles CHUNK_SIZE elements
    int numBlocks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

    // Radix (number of buckets) for this run
    size_t radix = (size_t)1 << radixBits;

    // Per-block histograms and offsets (size: numBlocks * radix)
    unsigned int *d_blockHist    = NULL;
    unsigned int *d_blockOffsets = NULL;
    CUDA_CHECK(cudaMalloc(&d_blockHist,    numBlocks * radix * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets, numBlocks * radix * sizeof(unsigned int)));

    // Host buffers for hist and offsets
    unsigned int *h_blockHist    = (unsigned int*)malloc(numBlocks * radix * sizeof(unsigned int));
    unsigned int *h_blockOffsets = (unsigned int*)malloc(numBlocks * radix * sizeof(unsigned int));
    unsigned int *h_bucketTotals = (unsigned int*)malloc(radix * sizeof(unsigned int));
    unsigned int *h_bucketBase   = (unsigned int*)malloc(radix * sizeof(unsigned int));

    if (!h_blockHist || !h_blockOffsets || !h_bucketTotals || !h_bucketBase) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    int numPasses = 32 / radixBits;
    uint32_t radixMask;
    if (radixBits == 32) {
        radixMask = 0xFFFFFFFFu;
    } else {
        radixMask = (uint32_t)(((uint64_t)1 << radixBits) - 1u);
    }

    uint32_t *d_in  = d_data;
    uint32_t *d_out = d_temp;

    for (int pass = 0; pass < numPasses; ++pass) {
        int shift = pass * radixBits;

        size_t shmemSize = radix * sizeof(unsigned int);

        // 1) Build per-block histograms on GPU
        buildBlockHistKernel<<<numBlocks, BLOCK_SIZE, shmemSize>>>(
            d_in, n, shift, d_blockHist, radix, radixMask
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2) Copy per-block histograms to host
        CUDA_CHECK(cudaMemcpy(h_blockHist, d_blockHist,
                              numBlocks * radix * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        // 3) Compute total counts per bucket
        for (size_t b = 0; b < radix; ++b) {
            h_bucketTotals[b] = 0;
        }
        for (int blk = 0; blk < numBlocks; ++blk) {
            size_t base = (size_t)blk * radix;
            for (size_t b = 0; b < radix; ++b) {
                h_bucketTotals[b] += h_blockHist[base + b];
            }
        }

        // 4) Compute global bucket bases (exclusive prefix sum on buckets)
        unsigned int sum = 0;
        for (size_t b = 0; b < radix; ++b) {
            h_bucketBase[b] = sum;
            sum += h_bucketTotals[b];
        }

        // 5) Compute per-block offsets for each bucket
        for (size_t b = 0; b < radix; ++b) {
            unsigned int running = h_bucketBase[b];
            for (int blk = 0; blk < numBlocks; ++blk) {
                size_t idx = (size_t)blk * radix + b;
                h_blockOffsets[idx] = running;
                running += h_blockHist[idx];
            }
        }

        // 6) Copy per-block offsets back to device
        CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_blockOffsets,
                              numBlocks * radix * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        // 7) Scatter into temp buffer in parallel
        scatterKernel<<<numBlocks, BLOCK_SIZE, shmemSize>>>(
            d_in, d_out, n, shift, d_blockOffsets, radix, radixMask
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 8) Swap d_in and d_out for next pass
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

    // If the final sorted data ended up in the temp buffer, copy it back
    if (d_in != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, d_in, n * sizeof(uint32_t),
                              cudaMemcpyDeviceToDevice));
    }

    // Free temporary GPU memory
    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_blockHist));
    CUDA_CHECK(cudaFree(d_blockOffsets));

    free(h_blockHist);
    free(h_blockOffsets);
    free(h_bucketTotals);
    free(h_bucketBase);
}

// ============================= main =============================

int main(int argc, char **argv)
{
    const int N = 1 << 28; // 33,554,432 elements

    if (argc != 3) {
        fprintf(stderr,
                "Usage: %s <radix_bits> <input_file>\n", argv[0]);
        return 1;
    }

    int radixBits = atoi(argv[1]);
    // Allow any radixBits in [1, 32] as long as it divides 32
    if (radixBits <= 0 || radixBits > 32 || (32 % radixBits) != 0) {
        fprintf(stderr,
                "Error: radix_bits must be > 0, <= 32, and divide 32 (got %d)\n",
                radixBits);
        return 1;
    }

    const char *inputPath = argv[2];

    uint32_t *h_in  = (uint32_t*)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Read N integers from the input text file (space/newline separated)
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
    radixSortGPU(d_data, N, radixBits, &elapsed_ms);

    printf("GPU radix sort time (%d-bit passes): %.3f ms\n",
           radixBits, elapsed_ms);

    // Copy back result to host and verify
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

