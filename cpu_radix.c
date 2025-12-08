#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define RADIX_BITS 8
#define RADIX (1 << RADIX_BITS)   // 256
#define RADIX_MASK (RADIX - 1)

// Simple 8-bit-per-pass LSD radix sort on CPU (32-bit unsigned ints)
void radixSortCPU8bit(uint32_t *data, int n) {
    if (n <= 1) return;

    uint32_t *temp = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!temp) {
        fprintf(stderr, "radixSortCPU8bit: malloc failed\n");
        exit(EXIT_FAILURE);
    }

    uint32_t *in  = data;
    uint32_t *out = temp;

    // 4 passes for 32-bit integers (8 bits per pass)
    for (int pass = 0; pass < 32 / RADIX_BITS; ++pass) {
        int shift = pass * RADIX_BITS;

        // 1) Build histogram
        uint32_t hist[RADIX] = {0};

        for (int i = 0; i < n; ++i) {
            uint32_t v = in[i];
            uint32_t bucket = (v >> shift) & RADIX_MASK;
            hist[bucket]++;
        }

        // 2) Build exclusive prefix sum -> offsets
        uint32_t offsets[RADIX];
        uint32_t sum = 0;
        for (int b = 0; b < RADIX; ++b) {
            offsets[b] = sum;
            sum += hist[b];
        }

        // 3) Stable scatter into out[]
        for (int i = 0; i < n; ++i) {
            uint32_t v = in[i];
            uint32_t bucket = (v >> shift) & RADIX_MASK;
            uint32_t pos = offsets[bucket]++;
            out[pos] = v;
        }

        // 4) Swap in/out for next pass
        uint32_t *tmp = in;
        in = out;
        out = tmp;
    }

    // After an even number of passes (4), data ends up in `in`.
    // If `in` is not the original `data` pointer, copy back.
    if (in != data) {
        memcpy(data, in, n * sizeof(uint32_t));
    }

    free(temp);
}

int main(void) {
    const int N = 1 << 25; // 1M elements, same scale as GPU test if you want
    uint32_t *arr  = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *copy = (uint32_t *)malloc(N * sizeof(uint32_t));

    if (!arr || !copy) {
        fprintf(stderr, "Host malloc failed\n");
        return 1;
    }

    // Fill with pseudo-random data
    for (int i = 0; i < N; ++i) {
        arr[i] = (uint32_t)rand();
    }

    // Keep a copy if you want to compare with some other sort later
    memcpy(copy, arr, N * sizeof(uint32_t));

    // Time the CPU radix sort
    clock_t start = clock();
    radixSortCPU8bit(arr, N);
    clock_t end = clock();

    double elapsed_ms = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;
    printf("CPU 8-bit-per-pass radix sort time: %.3f ms\n", elapsed_ms);

    // Verify sorted order
    int fail_idx = -1;
    for (int i = 1; i < N; ++i) {
        if (arr[i - 1] > arr[i]) {
            fail_idx = i;
            break;
        }
    }

    if (fail_idx == -1) {
        printf("Sort verification PASSED\n");
    } else {
        printf("Sort verification FAILED at index %d (values %u, %u)\n",
               fail_idx, arr[fail_idx - 1], arr[fail_idx]);
    }

    free(arr);
    free(copy);

    return 0;
}

