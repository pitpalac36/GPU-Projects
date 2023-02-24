#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "sha1.cuh"

#define CHUNK_SIZE 64
#define MAX_SUFFIX_LEN 5
#define SHA_BYTE_SIZE 20
#define alphabet "0123456789abcdefghijklmnopqrstuvwxyz"

typedef struct {
    char input[CHUNK_SIZE],
         suffix[MAX_SUFFIX_LEN];
    int len, suffix_len;
    bool *found;
} NONCE;

__device__ bool ends_with(const char* src, int src_len, const char *end, int end_len) {
    if (src_len < end_len) return false;
    for (int i = 0; i < end_len; ++i)
        if (src[src_len-i-1] != end[end_len-i-1]) return false;
    return true;
}

__device__ void set_suffix(NONCE *nonce, uint64_t id) {
        int alphabet_size = sizeof(alphabet) / sizeof(char);
        int k = 0;
        while (id > 0) {
                if (*nonce->found) break;
                nonce->input[nonce->len + k] = alphabet[id % alphabet_size];
                id /= alphabet_size;
                ++k;
        }
        nonce-> len += k;
}

__global__ void kernel(NONCE nonce) {
    if (*nonce.found) return; // some thread found the nonce
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    uint64_t id = blockId * blockDim.x + threadIdx.x;
    set_suffix(&nonce, id);

    unsigned char sha_result[SHA_BYTE_SIZE];
    SHA1(sha_result, (const unsigned char*)nonce.input, nonce.len);

    if (ends_with((char*)sha_result, SHA_BYTE_SIZE, nonce.suffix, nonce.suffix_len)) {
        *nonce.found = true;

        printf("SHA: ");
        for (int i=0; i < SHA_BYTE_SIZE; ++i)
            printf("%x", sha_result[i]);
        printf("\n");
        printf("NONCE: ");
        for (int i=0; i < nonce.len; ++i)
            printf("%c", nonce.input[i]);
        printf("\n");
    }
}

void read_input(char *out_str, int *out_len, const char *message, int max_len) {
        printf("%s", message);
        fgets(out_str, max_len+1, stdin);
        out_str[strcspn(out_str, "\r\n")] = 0;
        *out_len = strlen(out_str);
}

int main() {
    NONCE nonce;
    cudaMalloc(&nonce.found, sizeof(bool)); cudaMemset(nonce.found, 0, sizeof(bool));

    int max_input_len = CHUNK_SIZE - 8;

    read_input(nonce.input, &nonce.len, "Input:", max_input_len);
    read_input(nonce.suffix, &nonce.suffix_len, "SHA suffix:", MAX_SUFFIX_LEN);
    printf("Suffix hex: 0x");
    for (int i = 0; i < nonce.suffix_len; ++i)
        printf("%x", nonce.suffix[i]);
    printf("\n");

    cudaSetDevice(0);
    kernel<<<dim3(1024, 1024), 1024>>>(nonce);
    cudaDeviceSynchronize();
    cudaFree(nonce.found);
    return 0;
}

