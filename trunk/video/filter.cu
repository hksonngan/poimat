#include "filter.h"
#include <stdio.h>
#include <cufft.h>

#define BLOCK_SIZE   128
#define BLOCK_LEN    128
#define COLOR        4
#define div(n,m) (((n)+(m-1))/(m))

#define TEX_NUM 5
texture<uchar4, 2, cudaReadModeNormalizedFloat> texRef0;
texture<float4, 2, cudaReadModeElementType> texRef1;
texture<float4, 2, cudaReadModeElementType> texRef2;
texture<float, 2, cudaReadModeElementType> texRef3;
texture<float, 2, cudaReadModeElementType> texRef4;
cudaArray* cuArray[TEX_NUM];

#define i(x,y)  tex2D((texRef0),x,y)
#define f(x,y)  tex2D((texRef1),x,y)
#define b(x,y)  tex2D((texRef2),x,y)
#define da(x,y) tex2D((texRef3),x,y)
#define a(x,y)  tex2D((texRef4),x,y)

#define kernel \
    int by = blockIdx.y*BLOCK_LEN;\
    int x = blockIdx.x*BLOCK_SIZE+threadIdx.x;\
    if (x >= w) return;\
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

#define INFINITY 1E+37

__global__ void linearFill(float4 *data, float key, int w, int h)
{
    __shared__ float4 temp[BLOCK_SIZE*2];
    __shared__ float  dist[BLOCK_SIZE*2];

    int thid = threadIdx.x;
    int size = blockDim.x*2;
    int offset = 1;

    int x = blockIdx.x*size;
    int y = blockIdx.y;

    size = min(size, w-x);

    if (a(x+2*thid,y) == key) {
        temp[2*thid]   = i(x+2*thid,y);
        dist[2*thid]   = 0.0f;
    } else {
        temp[2*thid]   = float4();
        dist[2*thid]   = INFINITY;
    }
    if (a(x+2*thid+1,y) == key) {
        temp[2*thid+1] = i(x+2*thid+1,y);
        dist[2*thid+1] = 0.0f;
    } else {
        temp[2*thid+1] = float4();
        dist[2*thid+1] = INFINITY;
    }

#define relax(i,j,o) \
    if(dist[i] > dist[j]) {\
        temp[i] = temp[j];\
        dist[i] = dist[j]+o;\
    }

    for (int i = size/2; i > 0; i /= 2)
    {
        __syncthreads();
        if (thid < i)
        {
            int a0 = offset*(2*thid);
            int a1 = offset*(2*thid+1)-1;
            int b0 = offset*(2*thid+1);
            int b1 = offset*(2*thid+2)-1;
            relax(a1,b0,1.0f);
            relax(a0,b0,offset);
            relax(b0,a1,1.0f);
            relax(b1,a1,offset);
        }
        offset *= 2;
    }
    offset *= 2;
    for (int i = 1; i < size; i *= 2)
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
    if (2*thid < size)
        data[y*w+x+2*thid]   = temp[2*thid];
    if (2*thid+1 < size)
        data[y*w+x+2*thid+1] = temp[2*thid+1];
}

__global__ void nearestFill(float4 *data, float key, int w, int h)
{
    int s,t;
    kernel {
        for (int i=0; i<10; i++) {
            for (int j=-i; j<=i; j++) {
                s = x+j;
                if (s >= 0 && s < w) {
                    t = y-i;
                    if (t >= 0 && a(s,t) == key)
                        goto found;
                    t = y+i;
                    if (t < h && a(s,t) == key)
                        goto found;
                }
                t = y+j;
                if (t >= 0 && t < h) {
                    s = x-i;
                    if (s >= 0 && a(s,t) == key)
                        goto found;
                    s = x+i;
                    if (s < w && a(s,t) == key)
                        goto found;
                }
            }
        }
        continue;
found:
        data[y*w+x] = i(s,t);
    }
}

#define dot(p,q) (2.0f*(p.x*q.x)+4.0f*(p.y*q.y)+3.0f*(p.z*q.z))
#define norm(p) sqrt(dot(p,p))

__global__ void alphaFromTrimap(float *data, int w, int h)
{
    kernel {
        float fx = 2.0f*x/w-1.0f;
        float fy = 2.0f*y/h-1.0f;
        float fr = sqrt(fx*fx+fy*fy);
        if (fr < 0.333f) {
           data[y*w+x] = 0.0f;
        } else if (fr < 0.666f) {
           data[y*w+x] = 0.5f;
        } else {
           data[y*w+x] = 1.0f;
        }
    }
}

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
            data[y*w+x] = norm(divI)/(norm(deltaFB)+0.001f);
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
        /*float alpha = a(x,y);
        data[y*w+x].x = alpha;
        data[y*w+x].y = alpha;
        data[y*w+x].z = alpha;*/
        data[y*w+x] = f(x,y);
    }
}


//
//  interface
//

static bool initialized = false;
static float *alpha = NULL;
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
    cudaChannelFormatDesc channelDescF4 = cudaCreateChannelDesc<float4>();
    cudaChannelFormatDesc channelDescF = cudaCreateChannelDesc<float>();

    cudaMallocArray(&cuArray[0], &channelDescB4, w, h);
    cudaMallocArray(&cuArray[1], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[2], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[3], &channelDescF, w, h);
    cudaMallocArray(&cuArray[4], &channelDescF, w, h);

    cudaBindTextureToArray(texRef0, cuArray[0]);
    cudaBindTextureToArray(texRef1, cuArray[1]);
    cudaBindTextureToArray(texRef2, cuArray[2]);
    cudaBindTextureToArray(texRef3, cuArray[3]);
    cudaBindTextureToArray(texRef4, cuArray[4]);

    cudaMalloc((void**)&alpha, w*h*sizeof(float));
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

    for (int i=0; i<TEX_NUM; i++){
        cudaFreeArray(cuArray[i]);
    }

    cudaFree(alpha);
    cudaFree(image);

    initialized = false;
}

extern "C" void
poissonFilter(
    const unsigned char *h_src,
    float *d_dst,
    size_t w,
    size_t h
) {
    dim3 grid(div(w,BLOCK_SIZE), div(h,BLOCK_LEN));
    int threads = min(BLOCK_SIZE, w);

    // texture 0: image
    cudaMemcpyToArray(cuArray[0], 0, 0, h_src, w*h*sizeof(uchar4), cudaMemcpyHostToDevice);

    // texture 4: alpha
    alphaFromTrimap <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[4], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Foreground
     */

    // texture 1: forground image
    linearFill <<< dim3(div(w,BLOCK_SIZE*2), h), min(BLOCK_SIZE, w/2), 0 >>> (image, 1.0f, w, h);
    cudaMemcpyToArray(cuArray[1], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Background
     */

    // texture 2: background image
    nearestFill <<< grid, threads, 0 >>> (image, 0.0f, w, h);
    cudaMemcpyToArray(cuArray[2], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);

    /*
     *  Calculate Alpha
     */

    // texture 3: gradient alpha
    alphaGradient <<< grid, threads, 0 >>> (alpha, w, h);
    cudaMemcpyToArray(cuArray[3], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    alphaInitialize <<< grid, threads, 0 >>> (alpha, w, h);
    for (int i=0; i<0; i++){
        // texture 4: alpha
        cudaMemcpyToArray(cuArray[4], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
        alphaReconstruct <<< grid, threads, 0 >>> (alpha, w, h);
    }
    cudaMemcpyToArray(cuArray[4], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice);

    // output alpha
    alphaOutput <<< grid, threads, 0 >>> (image, w, h);
    cudaMemcpy(d_dst, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice);
}
