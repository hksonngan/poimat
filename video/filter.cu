#include "filter.h"
#include <stdio.h>

#define BLOCK_SIZE   128
#define BLOCK_LEN    64

#define BLOCK_MARGIN BLOCK_SIZE/2
#define BLOCK_ROW    BLOCK_SIZE*2

#define COLOR        4

typedef unsigned char type;
texture<type, 2, cudaReadModeElementType> texRef;

__global__ void median(
    unsigned char *d_dst,
    int w,
    int h,
    int r,
    int pitch,
    int color
) {
    __shared__ unsigned char rowp[BLOCK_ROW];
    __shared__ unsigned char rowq[BLOCK_ROW];

    int bx = blockIdx.x*BLOCK_SIZE;
    int by = blockIdx.y*BLOCK_LEN;
    int thid = threadIdx.x;

    short filter = 2*r+1;
    short half = filter*filter/2;
    short count;

    int offset = BLOCK_MARGIN+thid-r;
    int shift  = (filter+r-thid%filter)%filter;

    int ymin = max(by, r);
    int ymax = min(by+BLOCK_LEN, h-r);
    bool inside = bx+thid >= r && bx+thid < w-r;

    short h0[256];
    for (int i=0; i<256; i++) {
        h0[i] = 0;
    }

    int up = thid;
    int un = thid+BLOCK_SIZE;
    int xp = (bx-BLOCK_MARGIN+thid)*COLOR+color;
    int xn = (bx-BLOCK_MARGIN+thid+BLOCK_SIZE)*COLOR+color;

    for (int y=ymin-r; y<=ymin+r; y++)
    {
        __syncthreads();

        rowp[up] = tex2D(texRef, xp, y);
        rowp[un] = tex2D(texRef, xn, y);

        __syncthreads();

        for (int k=0; k<filter; k++) {
            int i = offset+(k+shift)%filter;
            h0[rowp[i]]++;
        }
    }

    short l;

    for (int y=ymin; y<ymax; y++)
    {
        count = half;
        for(l=0; count >= h0[l]; l++) {
            count -= h0[l];
        }

        __syncthreads();

        if (inside) {
            d_dst[((bx+thid)+y*pitch)*COLOR+color] = l;
        }

        rowp[up] = tex2D(texRef, xp, y+r+1);
        rowp[un] = tex2D(texRef, xn, y+r+1);
        rowq[up] = tex2D(texRef, xp, y-r);
        rowq[un] = tex2D(texRef, xn, y-r);

        __syncthreads();

        for (int k=0; k<filter; k++) {
            int i = offset+(k+shift)%filter;
            h0[rowp[i]]++;
            h0[rowq[i]]--;
        }

        __syncthreads();
    }
}

extern "C" void
medianFilter(
    const unsigned char *h_src,
    unsigned char *h_dst,
    unsigned int w,
    unsigned int h,
    unsigned int r
) {
    int size = w*h*COLOR;

    #define div(n,m) (((n)+(m-1))/(m))
    dim3 grid(div(w,BLOCK_SIZE), div(h,BLOCK_LEN));
    int threads = min(BLOCK_SIZE, w);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<type>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, w*COLOR, h);
    cudaMemcpyToArray(cuArray, 0, 0, h_src, size, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(texRef, cuArray);

    median <<< grid, threads, 0 >>> (h_dst, w, h, r, w, 0);
    median <<< grid, threads, 0 >>> (h_dst, w, h, r, w, 1);
    median <<< grid, threads, 0 >>> (h_dst, w, h, r, w, 2);

    cudaFreeArray(cuArray);
}
