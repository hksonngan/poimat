#include "kernel.h"
#include "alpha.h"
#include "flood.h"
#include "smooth.h"
#include "bilayer.h"
#include "filter.h"

#include <stdio.h>

#define cudaDebug(op) { \
    cudaError_t err = op; \
    printf("File %s Line %d\n\t%s(%d)\n", __FILE__, __LINE__, cudaGetErrorString(err), err); \
}
#define cudaCheck() cudaDebug(cudaGetLastError())
 
//////////////////////////////
//  Variables & Constants   //
//////////////////////////////

#define fillRadius   1024
#define blurRadius   0
#define trimapRadius 16

static int width;
static int height;
static int size;

static dim3 grid;
static int threads;
static dim3 gridX;
static int threadsX;
static dim3 gridY;
static int threadsY;

//  Host Memory
static float  *alpha      = NULL;
static float4 *image      = NULL;
static float2 *nearest    = NULL;
static int2   *histogram  = NULL;
static uchar  *trimap     = NULL;
static uchar  *temperal   = NULL;

//  DeviceMemory Memory
#define TEX_NUM 10
cudaArray* cuArray[TEX_NUM];

#define arrImage      cuArray[0]
#define arrTrimap     cuArray[1]
#define arrTemperal   cuArray[9]
#define arrHistogram  cuArray[2]
#define arrTemp       cuArray[4]
#define arrForeground cuArray[5]
#define arrBackground cuArray[6]
#define arrNearest    cuArray[3]
#define arrLapAlpha   cuArray[7]
#define arrAlpha      cuArray[8]

//////////////////////////////
//  Algorithms              //
//////////////////////////////

/*
    Function Fill
        - featuring jump flood algorithm for voronoi diagram

    Input:  arrImage, arrAlpha, key alpha
    Using:  nearest, arrNearest
    Output: image
*/

inline void Fill(float key, int radius)
{
    floodInitialize <<< grid, threads, 0 >>> (image, nearest, key, width, height);
    for (int step=radius; step>0; step/=2) {
        cudaMemcpyToArray(arrNearest, 0, 0, nearest, size*sizeof(float2), cudaMemcpyDeviceToDevice);
        floodStep <<< grid, threads, 0 >>> (nearest, key, step, width, height);
    }
    cudaMemcpyToArray(arrNearest, 0, 0, nearest, size*sizeof(float2), cudaMemcpyDeviceToDevice);
    floodFinalize <<< grid, threads, 0 >>> (image, key, radius, width, height);
}

/*
    Function: Blur
        - featuring shrink-half averaging radius

    Input:  image, arrNearest
    Using:  arrTemp
    Output: image
*/

inline void Blur(int radius)
{
    for (int i=radius; i>1; i/=2) {
        cudaMemcpyToArray(arrTemp, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);
        smoothScan <<< gridX, threadsX, 0 >>> (image, true, width, height);
        cudaMemcpyToArray(arrTemp, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);
        smoothScan <<< gridY, threadsY, 0 >>> (image, false, width, height);
        cudaMemcpyToArray(arrTemp, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);
        smoothAverage <<< grid, threads, 0 >>> (image, i, width, height);
    }
}

/*
    Function Solver
        - featuring Jacobi method for linear system solution

    Input:  arrImage, arrForeground, arrBackground
    Using:  arrLapAlpha
    Output: arrAlpha
*/

inline void Solver()
{
    alphaLaplacian <<< grid, threads, 0 >>> (alpha, width, height);
    cudaMemcpyToArray(arrLapAlpha, 0, 0, alpha, size*sizeof(float), cudaMemcpyDeviceToDevice);
    alphaInitialize <<< grid, threads, 0 >>> (alpha, width, height);
    for (int i=0; i<64; i++){
        cudaMemcpyToArray(arrAlpha, 0, 0, alpha, size*sizeof(float), cudaMemcpyDeviceToDevice);
        alphaReconstruct <<< grid, threads, 0 >>> (alpha, width, height);
    }
    alphaRefinement <<< grid, threads, 0 >>> (alpha, width, height);
    cudaMemcpyToArray(arrAlpha, 0, 0, alpha, size*sizeof(float), cudaMemcpyDeviceToDevice);
}

/*
    Function Segment
        - featuring graph cut for bilayer segmentation / jump flood algorithm for the third region

    Input:  histogram, arrImage
    Using:  image, arrHistogram, arrNearest, arrTemp
    Output: trimap
*/

inline void Segment(int* f_point, int f_count, int* b_point, int b_count)
{
    cudaMemcpyToArray(arrHistogram, 0, 0, histogram, HIST_SIZE*sizeof(int2), cudaMemcpyDeviceToDevice);
    graphConstruct <<< grid, threads, 0 >>> (image, width, height);
    cudaMemcpyToArray(arrTemp, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);
    for (int i=0; i<8; i++) {
        graphCut <<< grid, threads, 0 >>> (alpha, width, height);
        cudaMemcpyToArray(arrAlpha, 0, 0, alpha, size*sizeof(float), cudaMemcpyDeviceToDevice);
    }

    trimapInitialize <<< grid, threads, 0 >>> (nearest, width, height);
	trimapPoint <<< div(f_count, BLOCK_SIZE), min(f_count, BLOCK_SIZE), 0 >>> ((int2*)f_point, f_count, 1.0f, alpha, nearest, width, height);
	trimapPoint <<< div(f_count, BLOCK_SIZE), min(f_count, BLOCK_SIZE), 0 >>> ((int2*)f_point, f_count, 0.0f, alpha, nearest, width, height);
    for (int step=trimapRadius; step>0; step/=2) {
        cudaMemcpyToArray(arrNearest, 0, 0, nearest, size*sizeof(float2), cudaMemcpyDeviceToDevice);
        floodStep <<< grid, threads, 0 >>> (nearest, 0.5f, step, width, height);
    }
    cudaMemcpyToArray(arrNearest, 0, 0, nearest, size*sizeof(float2), cudaMemcpyDeviceToDevice);
    trimapOutput <<< grid, threads, 0 >>> (trimap, trimapRadius, width, height);
}

//////////////////////////////
//  Interfaces              //
//////////////////////////////

void initializeFilter(
    unsigned int w,
    unsigned int h)
{
    width  = w;
    height = h;
    size   = w*h;

    grid     = dim3(div(w,BLOCK_SIZE), div(h,BLOCK_LEN));
    threads  = min(BLOCK_SIZE, w);
    gridX    = dim3(div(w,BLOCK_SIZE*2), h);
    threadsX = min(BLOCK_SIZE, (w+1)/2);
    gridY    = dim3(div(h,BLOCK_SIZE*2), w);
    threadsY = min(BLOCK_SIZE, (h+1)/2);

    cudaMallocArray(&arrImage,      &cudaCreateChannelDesc<uchar4>(), w, h);
    cudaMallocArray(&arrTrimap,     &cudaCreateChannelDesc<uchar>(),  w, h);
    cudaMallocArray(&arrTemperal,   &cudaCreateChannelDesc<uchar>(),  w, h);
    cudaMallocArray(&arrHistogram,  &cudaCreateChannelDesc<int2>(), HIST_SIZE, 1);
    cudaMallocArray(&arrTemp,       &cudaCreateChannelDesc<float4>(), w, h);
    cudaMallocArray(&arrForeground, &cudaCreateChannelDesc<float4>(), w, h);
    cudaMallocArray(&arrBackground, &cudaCreateChannelDesc<float4>(), w, h);
    cudaMallocArray(&arrNearest,    &cudaCreateChannelDesc<float2>(), w, h);
    cudaMallocArray(&arrLapAlpha,   &cudaCreateChannelDesc<float>(),  w, h);
    cudaMallocArray(&arrAlpha,      &cudaCreateChannelDesc<float>(),  w, h);

    bindFloodTexture  (arrImage, arrNearest, arrAlpha);
    bindAlphaTexture  (arrImage, arrTrimap, arrForeground, arrBackground, arrLapAlpha, arrAlpha);
    bindSmoothTexture (arrImage, arrTemp, arrNearest);
    bindBilayerTexture(arrImage, arrTemperal, arrHistogram, arrNearest, arrTemp, arrAlpha);

    cudaMalloc((void**)&alpha,      w*h*sizeof(float));
    cudaMalloc((void**)&image,      w*h*sizeof(float4));
    cudaMalloc((void**)&nearest,    w*h*sizeof(float2));
    cudaMalloc((void**)&histogram,  HIST_SIZE*sizeof(int2));
    cudaMalloc((void**)&trimap,     w*h*sizeof(uchar));
    cudaMalloc((void**)&temperal,   w*h*sizeof(uchar));

    cudaMemset(histogram, 0, HIST_SIZE*sizeof(int2));
}

void releaseFilter()
{
    unbindFloodTexture();
    unbindAlphaTexture();
    unbindSmoothTexture();
    unbindBilayerTexture();

    for (int i=0; i<TEX_NUM; i++) {
        cudaFreeArray(cuArray[i]);
    }

    cudaFree(alpha);
    cudaFree(nearest);
    cudaFree(image);
    cudaFree(histogram);
    cudaFree(trimap);
    cudaFree(temperal);
}

void poissonFilter(
    const unsigned char *h_src,
    const unsigned char *h_trimap,
    float *d_dst)
{
    cudaMemcpyToArray(arrImage, 0, 0, h_src, size*sizeof(uchar4), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(arrTrimap, 0, 0, h_trimap, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

    alphaFromTrimap <<< grid, threads, 0 >>> (alpha, width, height);
    cudaMemcpyToArray(arrAlpha, 0, 0, alpha, size*sizeof(float), cudaMemcpyDeviceToDevice);


    for (int time=0; time<2; time++)
    {
        //////////////////////////////
        //  Calculate Foreground
        Fill(1.0f, fillRadius);
        Blur(blurRadius);
        cudaMemcpyToArray(arrForeground, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);

        //////////////////////////////
        // Calculate Background
        Fill(0.0f, fillRadius);
        Blur(blurRadius);
        cudaMemcpyToArray(arrBackground, 0, 0, image, size*sizeof(float4), cudaMemcpyDeviceToDevice);

        //////////////////////////////
        // Calculate Alpha
        Solver();
    }

    //////////////////////////////
    // Output Alpha
    alphaOutput <<< grid, threads, 0 >>> ((float4*)d_dst, width, height);

    //////////////////////////////
    // Calculate Histogram
    cudaMemcpyToArray(arrTemperal, 0, 0, temperal, size*sizeof(uchar), cudaMemcpyDeviceToDevice);
    graphStatistic <<< grid, threads, 0 >>> (histogram, temperal, width, height);
    graphIterate <<< div(HIST_SIZE, BLOCK_SIZE), BLOCK_SIZE, 0 >>> (histogram);
}

void trimapFilter(
    const unsigned char *h_src,
    unsigned char *h_trimap,
	int* f_point, int f_count,
	int* b_point, int b_count)
{
    cudaMemcpyToArray(arrImage, 0, 0, h_src, size*sizeof(uchar4), cudaMemcpyHostToDevice);

    //////////////////////////////
    // Calculate Trimap
    Segment(f_point, f_count, b_point, b_count);

    cudaMemcpy(h_trimap, trimap, size*sizeof(uchar), cudaMemcpyDeviceToHost);
}
