#include "filter.h"
#include "kernel.h"
#include "alpha.h"
#include "flood.h"

#define TEX_NUM 8
cudaArray* cuArray[TEX_NUM];

//
//  interface
//

static bool initialized = false;
static float *alpha = NULL;
static float2 *nearest = NULL;
static float4 *image = NULL;

extern "C" void
initializeTexture(
    unsigned int w,
    unsigned int h
) {
    if (initialized) {
        releaseTexture();
    }

    cudaChannelFormatDesc channelDescB4 = cudaCreateChannelDesc<uchar4>();
    cudaChannelFormatDesc channelDescB = cudaCreateChannelDesc<unsigned char>();
    cudaChannelFormatDesc channelDescF4 = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc channelDescF2 = cudaCreateChannelDesc<float2>();
    cudaChannelFormatDesc channelDescF = cudaCreateChannelDesc<float>();

    cudaMallocArray(&cuArray[0], &channelDescB4, w, h); // image
    cudaMallocArray(&cuArray[1], &channelDescB,  w, h); // trimap
    cudaMallocArray(&cuArray[2], &channelDescF4, w, h); // temp image
    cudaMallocArray(&cuArray[3], &channelDescF4, w, h); // foreground
    cudaMallocArray(&cuArray[4], &channelDescF4, w, h); // background
    cudaMallocArray(&cuArray[5], &channelDescF2, w, h); // nearest pixel
    cudaMallocArray(&cuArray[6], &channelDescF,  w, h); // laplacian alpha
    cudaMallocArray(&cuArray[7], &channelDescF,  w, h); // alpha

    bindFloodTexture(cuArray[0], cuArray[7], cuArray[5]);
    bindAlphaTexture(cuArray[0], cuArray[1], cuArray[3], cuArray[4], cuArray[6], cuArray[7]);

    cudaMalloc((void**)&alpha,   w*h*sizeof(float));
    cudaMalloc((void**)&nearest, w*h*sizeof(float2));
    cudaMalloc((void**)&image,   w*h*sizeof(float4));

    initialized = true;
}

extern "C" void
releaseTexture(
) {
    if (!initialized) {
        return;
    }

    unbindFloodTexture();
    unbindAlphaTexture();

    for (int i=0; i<TEX_NUM; i++){
        cudaFreeArray(cuArray[i]);
    }

    cudaFree(alpha);
    cudaFree(nearest);
    cudaFree(image);

    initialized = false;
}

extern "C" void
poissonFilter(
    const unsigned char *h_src,
    const unsigned char *h_trimap,
    float *d_dst,
    size_t w,
    size_t h
) {
    dim3 grid(div(w,BLOCK_SIZE), div(h,BLOCK_LEN));
    int threads = min(BLOCK_SIZE, w);
    int maxJump;
    for (maxJump=1; maxJump<w || maxJump<h; maxJump*=2);

    // cuda array 0: image
    cudaMemcpyToArray(cuArray[0], 0, 0, h_src, w*h*sizeof(uchar4), cudaMemcpyHostToDevice);

    // cuda array 1: trimap
    cudaMemcpyToArray(cuArray[1], 0, 0, h_trimap, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // cuda array 7: alpha
    alphaFromTrimap <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[7], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    for (int time=0; time<4; time++) {

    /*
     *  Calculate Foreground
     */

    preflood  <<< grid, threads, 0 >>> (image, nearest, 1.0f, w, h);
    for (int step=maxJump; step>0; step/=2) {
        // cuda array 5: nearest pixel
        cudaMemcpyToArray(cuArray[5], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice);
        flood <<< grid, threads, 0 >>> (nearest, step, w, h);
    }
    // cuda array 5: nearest pixel
    cudaMemcpyToArray(cuArray[5], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice);
    postflood <<< grid, threads, 0 >>> (image, w, h);
    // cuda array 3: forground image
    cudaMemcpyToArray(cuArray[3], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Background
     */

    preflood  <<< grid, threads, 0 >>> (image, nearest, 0.0f, w, h);
    for (int step=maxJump; step>0; step/=2) {
        // cuda array 5: nearest pixel
        cudaMemcpyToArray(cuArray[5], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice);
        flood <<< grid, threads, 0 >>> (nearest, step, w, h);
    }
    // cuda array 5: nearest pixel
    cudaMemcpyToArray(cuArray[5], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice);
    postflood <<< grid, threads, 0 >>> (image, w, h);
    // cuda array 4: background image
    cudaMemcpyToArray(cuArray[4], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Alpha
     */

    // texture 6: laplacian alpha
    alphaGradient <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[6], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
    alphaInitialize <<< grid, threads, 0 >>> (alpha, w, h);
    for (int i=0; i<64; i++){
        // texture 7: alpha
        cudaMemcpyToArray(cuArray[7], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
        alphaReconstruct <<< grid, threads, 0 >>> (alpha, w, h);
    }
    alphaRefinement <<< grid, threads, 0 >>> (alpha, w, h);
    // texture 7: alpha
    cudaMemcpyToArray(cuArray[7], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    }

    // output alpha
    alphaOutput <<< grid, threads, 0 >>> ((float4*)d_dst, w, h);
}
