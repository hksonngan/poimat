#include "filter.h"
#include <stdio.h>
#include <cufft.h>

#define BLOCK_SIZE   128
#define BLOCK_LEN    128
#define COLOR        4
#define div(n,m) (((n)+(m-1))/(m))

#define TEX_NUM 6
texture<uchar4, 2, cudaReadModeNormalizedFloat> texRef0;
texture<unsigned char, 2, cudaReadModeNormalizedFloat> texRef1;
texture<float4, 2, cudaReadModeElementType> texRef2;
texture<float4, 2, cudaReadModeElementType> texRef3;
texture<float, 2, cudaReadModeElementType> texRef4;
texture<float, 2, cudaReadModeElementType> texRef5;
cudaArray* cuArray[TEX_NUM];

#define i(x,y)  tex2D((texRef0),x,y)
#define t(x,y)  tex2D((texRef1),x,y)
#define f(x,y)  tex2D((texRef2),x,y)
#define b(x,y)  tex2D((texRef3),x,y)
#define da(x,y) tex2D((texRef4),x,y)
#define a(x,y)  tex2D((texRef5),x,y)

#define kernel \
    int by = blockIdx.y*BLOCK_LEN;\
    int x = blockIdx.x*BLOCK_SIZE+threadIdx.x;\
    if (x >= w) return;\
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

#define INFINITY 1E+37

__global__ void pointFill(float4 *data, float *dist, float key, int w, int h)
{
    kernel {
        if (a(x,y) == key) {
            data[y*w+x] = i(x,y);
            dist[y*w+x] = 0.0f;
        } else {
            data[y*w+x] = float4();
            dist[y*w+x] = INFINITY;
        }
    }
}

#define relax(i,j,o) \
    if(sdist[i] > sdist[j]+o) {\
        sdata[i] = sdata[j];\
        sdist[i] = sdist[j]+o;\
    }

__global__ void linearFill(float4 *data, float *dist, int w, int h, bool direction)
{
    __shared__ float4 sdata[BLOCK_SIZE*2];
    __shared__ float  sdist[BLOCK_SIZE*2];

    int thid = threadIdx.x;
    int size = blockDim.x*2;
    int offset = 1;

    int x = blockIdx.x;
    int y = blockIdx.y;
    int t0, t1;

    if (direction == true) {
        x *= size;
        size = min(size, w-x);
        t0 = y*w+x+2*thid;
        t1 = y*w+x+2*thid+1;
    } else {
        y *= size;
        size = min(size, h-y);
        t0 = (y+2*thid)*w+x;
        t1 = (y+2*thid+1)*w+x;
    }

    if (2*thid < size) {
        sdata[2*thid]   = data[t0];
        sdist[2*thid]   = dist[t0];
    }
    if (2*thid+1 < size) {
        sdata[2*thid+1] = data[t1];
        sdist[2*thid+1] = dist[t1];
    }

    for (int i=size/2; i>0; i/=2)
    {
        __syncthreads();
        if (thid < i)
        {
            int a0 = offset*(2*thid);
            int a1 = offset*(2*thid+1)-1;
            int b0 = offset*(2*thid+1);
            int b1 = offset*(2*thid+2)-1;
            relax(a0,b0,offset);
            relax(a1,b0,1.0f);
            relax(b0,a1,1.0f);
            relax(b1,a1,offset);
        }
        offset *= 2;
    }

    for (int i=1; i<size/2; i*=2)
    {
        offset /= 2;
        __syncthreads();
        if (thid < i)
        {
            int a0 = offset*(2*thid);
            int a1 = offset*(2*thid+1)-1;
            int b0 = offset*(2*thid+1);
            int b1 = offset*(2*thid+2)-1;
            if (a1 < size) {
                relax(a1,a0,offset-1.0f);
                relax(a0,a1,offset-1.0f);
            }
            if (b1 < size) {
                relax(b1,b0,offset-1.0f);
                relax(b0,b1,offset-1.0f);
            }
            if (b0 < size) {
                relax(a1,b0,1.0f);
                relax(b0,a1,1.0f);
            }
        }
    }

    __syncthreads();
    if (2*thid < size) {
        data[t0]   = sdata[2*thid];
        dist[t0]   = sdist[2*thid];
    }
    if (2*thid+1 < size) {
        data[t1] = sdata[2*thid+1];
        dist[t1] = sdist[2*thid+1];
    }
}

__global__ void alphaFromTrimap(float *data, int w, int h)
{
    kernel {
        data[y*w+x] = t(x,y);
    }
}

#define dot(p,q) (2.0f*(p.x*q.x)+4.0f*(p.y*q.y)+3.0f*(p.z*q.z))
#define norm(p) sqrt(dot(p,p))

__global__ void alphaGradient(float *data, int w, int h)
{
    kernel {
        if (a(x,y) != 0.0f && a(x,y) != 1.0f) {
            float4 divI;
            divI.x = i(x-1,y).x+i(x+1,y).x+i(x,y-1).x+i(x,y+1).x-4.0f*i(x,y).x;
            divI.y = i(x-1,y).y+i(x+1,y).y+i(x,y-1).y+i(x,y+1).y-4.0f*i(x,y).y;
            divI.z = i(x-1,y).z+i(x+1,y).z+i(x,y-1).z+i(x,y+1).z-4.0f*i(x,y).z;
            float4 deltaFB;
            deltaFB.x = f(x,y).x-b(x,y).x;
            deltaFB.y = f(x,y).y-b(x,y).y;
            deltaFB.z = f(x,y).z-b(x,y).z;
            data[y*w+x] = dot(divI,deltaFB)/(dot(deltaFB,deltaFB)+0.001f);
        } else {
            data[y*w+x] = 0.0f;
        }
    }
}

__global__ void alphaInitialize(float *data, int w, int h)
{
    kernel {
        if (a(x,y) == 0.0f) {
            data[y*w+x] = 0.0f;
        } else if (a(x,y) == 1.0f) {
            data[y*w+x] = 1.0f;
        } else {
            float4 deltaIB;
            deltaIB.x = i(x,y).x-b(x,y).x;
            deltaIB.y = i(x,y).y-b(x,y).y;
            deltaIB.z = i(x,y).z-b(x,y).z;
            float4 deltaFB;
            deltaFB.x = f(x,y).x-b(x,y).x;
            deltaFB.y = f(x,y).y-b(x,y).y;
            deltaFB.z = f(x,y).z-b(x,y).z;
            float numerator = dot(deltaIB,deltaFB);
            float denominator = dot(deltaFB,deltaFB);
            if (denominator != 0.0f)
                data[y*w+x] = numerator/denominator;
            else
                data[y*w+x] = 0.0f;
        }
    }
}

__global__ void alphaReconstruct(float *data, int w, int h)
{
    kernel {
        if (a(x,y) != 0.0f && a(x,y) != 1.0f) {
            data[y*w+x] = 0.25f*(a(x-1,y)+a(x+1,y)+a(x,y-1)+a(x,y+1)-da(x,y));
        }
    }
}

__global__ void alphaOutput(float4 *data, int w, int h)
{
    kernel {
        float alpha = max(0.0f, min(1.0f, a(x,y)));
        data[y*w+x].x = f(x,y).x*alpha;
        data[y*w+x].y = f(x,y).y*alpha;
        data[y*w+x].z = f(x,y).z*alpha;
    }
}


//
//  interface
//

static bool initialized = false;
static float *alpha = NULL;
static float *distance = NULL;
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
    cudaChannelFormatDesc channelDescF = cudaCreateChannelDesc<float>();

    cudaMallocArray(&cuArray[0], &channelDescB4, w, h);
    cudaMallocArray(&cuArray[1], &channelDescB, w, h);
    cudaMallocArray(&cuArray[2], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[3], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[4], &channelDescF, w, h);
    cudaMallocArray(&cuArray[5], &channelDescF, w, h);

    cudaBindTextureToArray(texRef0, cuArray[0]);
    cudaBindTextureToArray(texRef1, cuArray[1]);
    cudaBindTextureToArray(texRef2, cuArray[2]);
    cudaBindTextureToArray(texRef3, cuArray[3]);
    cudaBindTextureToArray(texRef4, cuArray[4]);
    cudaBindTextureToArray(texRef5, cuArray[5]);

    cudaMalloc((void**)&alpha, w*h*sizeof(float));
    cudaMalloc((void**)&distance, w*h*sizeof(float));
    cudaMalloc((void**)&image, w*h*sizeof(float4));

    initialized = true;
}

extern "C" void
releaseTexture(
) {
    if (!initialized) {
        return;
    }

    cudaUnbindTexture(texRef0);
    cudaUnbindTexture(texRef1);
    cudaUnbindTexture(texRef2);
    cudaUnbindTexture(texRef3);
    cudaUnbindTexture(texRef4);
    cudaUnbindTexture(texRef5);

    for (int i=0; i<TEX_NUM; i++){
        cudaFreeArray(cuArray[i]);
    }

    cudaFree(alpha);
    cudaFree(distance);
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

    dim3 gridX(div(w,BLOCK_SIZE*2), h);
    int threadsX = min(BLOCK_SIZE, (w+1)/2);

    dim3 gridY(w, div(h,BLOCK_SIZE*2));
    int threadsY = min(BLOCK_SIZE, (w+1)/2);

    // texture 0: image
    cudaMemcpyToArray(cuArray[0], 0, 0, h_src, w*h*sizeof(uchar4), cudaMemcpyHostToDevice);

    // texture 1: trimap
    cudaMemcpyToArray(cuArray[1], 0, 0, h_trimap, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // texture 4: alpha
    alphaFromTrimap <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Foreground
     */

    // texture 2: forground image
    pointFill  <<< grid, threads, 0 >>> (image, distance, 1.0f, w, h);
    linearFill <<< gridX, threadsX, 0 >>> (image, distance, w, h, true);
    linearFill <<< gridY, threadsY, 0 >>> (image, distance, w, h, false);
    cudaMemcpyToArray(cuArray[2], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Background
     */

    // texture 3: background image
    pointFill  <<< grid, threads, 0 >>> (image, distance, 0.0f, w, h);
    linearFill <<< gridX, threadsX, 0 >>> (image, distance, w, h, true);
    linearFill <<< gridY, threadsY, 0 >>> (image, distance, w, h, false);
    cudaMemcpyToArray(cuArray[3], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Alpha
     */

    // texture 4: gradient alpha
    alphaGradient <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[4], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    alphaInitialize <<< grid, threads, 0 >>> (alpha, w, h);
    for (int i=0; i<100; i++){
        // texture 5: alpha
        cudaMemcpyToArray(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
        alphaReconstruct <<< grid, threads, 0 >>> (alpha, w, h);
    }
    cudaMemcpyToArray(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    // output alpha
    alphaOutput <<< grid, threads, 0 >>> (image, w, h);
    cudaMemcpy(d_dst, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);
}
