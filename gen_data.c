#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <output_file>\n", argv[0]);
        return 1;
    }

    const char *outPath = argv[1];
    const int N = 1 << 28;  // 268,435,456 integers

    FILE *f = fopen(outPath, "w");
    if (!f) {
        perror("Error opening output file");
        return 1;
    }

    // Optional: better randomness
    srand(12345);

    for (int i = 0; i < N; i++) {
        uint32_t v = (uint32_t)rand();
        // Write integer followed by a space
        fprintf(f, "%u ", v);
    }

    fclose(f);
    return 0;
}

