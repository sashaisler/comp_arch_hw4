#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define THREADS_PER_BLOCK 256

// Kernel: compute bit and isZero arrays for this pass
__global__
void computeFlagsKernel(const uint32_t* d_in,
                        int n,
                        int shift,
                        uint8_t* d_isZero,
                        uint8_t* d_bit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t v = d_in[idx];
    uint8_t b = (v >> shift) & 1u;
    d_bit[idx] = b;
    d_isZero[idx] = (b == 0) ? 1u : 0u;
}

// Kernel: scatter based on bit, isZero scan (zeroPos), and totalZeros
__global__
void scatterKernel(const uint32_t* d_in,
                   uint32_t* d_out,
                   int n,
                   int shift,
                   const uint8_t* d_bit,
                   const uint8_t* d_isZero,
                   const uint32_t* d_zeroPos,
                   uint32_t totalZeros)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t v = d_in[idx];
    uint8_t  b = d_bit[idx];
    uint32_t zp = d_zeroPos[idx];

    uint32_t pos;
    if (b == 0) {
        // goes into the zero region at index zp
        pos = zp;
    } else {
        // number of ones before idx = idx - zeroPos[idx]
        uint32_t onesBefore = idx - zp;
        pos = totalZeros + onesBefore;
    }

    d_out[pos] = v;
}

// ---- Radix sort: binary LSD, stable, using Thrust for scan ----

void radixSortBinaryLSD(uint32_t* d_data, int n, float* elapsed_ms_out)
{
    uint32_t* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(uint32_t)));

    // Bit / isZero arrays (uint8_t is enough: 0 or 1)
    uint8_t* d_bit     = nullptr;
    uint8_t* d_isZero  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bit,    n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_isZero, n * sizeof(uint8_t)));

    // zeroPos (exclusive scan result)
    thrust::device_vector<uint32_t> zeroPos(n);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // For timing the whole sort
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int shift = 0; shift < 32; ++shift) {
        // 1) Compute bit and isZero
        computeFlagsKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, n, shift, d_isZero, d_bit
        );
        CUDA_CHECK(cudaGetLastError());

        // 2) Exclusive scan over isZero -> zeroPos
        // Thrust works on iterators. We cast d_isZero (uint8_t) to uint32_t by transform.
        // Simpler: copy to a thrust vector<uint32_t> first.

        // Temporary vector for scan input
        thrust::device_vector<uint32_t> isZeroInt(n);
        thrust::transform(
            thrust::device_pointer_cast(d_isZero),
            thrust::device_pointer_cast(d_isZero) + n,
            isZeroInt.begin(),
            [] __device__ (uint8_t x) { return (uint32_t)x; }
        );

        thrust::exclusive_scan(isZeroInt.begin(), isZeroInt.end(), zeroPos.begin());

        // 3) Compute totalZeros = zeroPos[n-1] + isZero[n-1]
        uint32_t lastZeroPos = zeroPos[n - 1];
        uint8_t  lastIsZero;
        CUDA_CHECK(cudaMemcpy(&lastIsZero, d_isZero + (n - 1),
                              sizeof(uint8_t), cudaMemcpyDeviceToHost));
        uint32_t totalZeros = lastZeroPos + (uint32_t)lastIsZero;

        // 4) Scatter into d_temp
        scatterKernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data,
            d_temp,
            n,
            shift,
            d_bit,
            d_isZero,
            thrust::raw_pointer_cast(zeroPos.data()),
            totalZeros
        );
        CUDA_CHECK(cudaGetLastError());

        // 5) Swap d_data and d_temp for next pass
        uint32_t* tmp = d_data;
        d_data = d_temp;
        d_temp = tmp;
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    if (elapsed_ms_out) *elapsed_ms_out = elapsed_ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_bit));
    CUDA_CHECK(cudaFree(d_isZero));
}

// ---------------------- main ------------------------

int main(void)
{
    const int N = 1 << 20; // 1M elements
    uint32_t* h_in = (uint32_t*)malloc(N * sizeof(uint32_t));
    if (!h_in) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Fill with random data
    for (int i = 0; i < N; ++i) {
        h_in[i] = (uint32_t)rand();
    }

    uint32_t* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_in, N * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    float elapsed_ms = 0.0f;
    radixSortBinaryLSD(d_data, N, &elapsed_ms);
    printf("GPU binary LSD radix sort time: %.3f ms\n", elapsed_ms);

    // Copy back and verify
    uint32_t* h_out = (uint32_t*)malloc(N * sizeof(uint32_t));
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

