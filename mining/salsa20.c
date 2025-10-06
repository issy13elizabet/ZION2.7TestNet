/*
 * Optimized Salsa20/8 implementation for ZION Yescrypt
 * Based on original Salsa20 by Daniel J. Bernstein
 */

#include <stdint.h>
#include <string.h>

#define ROTL(a,b) (((a) << (b)) | ((a) >> (32 - (b))))

void salsa20_8_core(uint32_t output[16], const uint32_t input[16])
{
    uint32_t x[16];
    int i;
    
    for (i = 0; i < 16; ++i) {
        x[i] = input[i];
    }
    
    // 8 rounds (4 double rounds)
    for (i = 8; i > 0; i -= 2) {
        x[ 4] ^= ROTL(x[ 0]+x[12], 7);
        x[ 8] ^= ROTL(x[ 4]+x[ 0], 9);
        x[12] ^= ROTL(x[ 8]+x[ 4],13);
        x[ 0] ^= ROTL(x[12]+x[ 8],18);
        
        x[ 9] ^= ROTL(x[ 5]+x[ 1], 7);
        x[13] ^= ROTL(x[ 9]+x[ 5], 9);
        x[ 1] ^= ROTL(x[13]+x[ 9],13);
        x[ 5] ^= ROTL(x[ 1]+x[13],18);
        
        x[14] ^= ROTL(x[10]+x[ 6], 7);
        x[ 2] ^= ROTL(x[14]+x[10], 9);
        x[ 6] ^= ROTL(x[ 2]+x[14],13);
        x[10] ^= ROTL(x[ 6]+x[ 2],18);
        
        x[ 3] ^= ROTL(x[15]+x[11], 7);
        x[ 7] ^= ROTL(x[ 3]+x[15], 9);
        x[11] ^= ROTL(x[ 7]+x[ 3],13);
        x[15] ^= ROTL(x[11]+x[ 7],18);
        
        x[ 1] ^= ROTL(x[ 0]+x[ 3], 7);
        x[ 2] ^= ROTL(x[ 1]+x[ 0], 9);
        x[ 3] ^= ROTL(x[ 2]+x[ 1],13);
        x[ 0] ^= ROTL(x[ 3]+x[ 2],18);
        
        x[ 6] ^= ROTL(x[ 5]+x[ 4], 7);
        x[ 7] ^= ROTL(x[ 6]+x[ 5], 9);
        x[ 4] ^= ROTL(x[ 7]+x[ 6],13);
        x[ 5] ^= ROTL(x[ 4]+x[ 7],18);
        
        x[11] ^= ROTL(x[10]+x[ 9], 7);
        x[ 8] ^= ROTL(x[11]+x[10], 9);
        x[ 9] ^= ROTL(x[ 8]+x[11],13);
        x[10] ^= ROTL(x[ 9]+x[ 8],18);
        
        x[12] ^= ROTL(x[15]+x[14], 7);
        x[13] ^= ROTL(x[12]+x[15], 9);
        x[14] ^= ROTL(x[13]+x[12],13);
        x[15] ^= ROTL(x[14]+x[13],18);
    }
    
    for (i = 0; i < 16; ++i) {
        output[i] = x[i] + input[i];
    }
}