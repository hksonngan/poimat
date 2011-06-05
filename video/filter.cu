#include "filter.h"

#define BLOCK_SIZE   128
#define BLOCK_LEN    128
#define COLOR        4

// int macro
#define div(n,m) (((n)+(m-1))/(m))
// color macro
#define dot(p,q) (2.0f*(p.x*q.x)+4.0f*(p.y*q.y)+3.0f*(p.z*q.z))
#define norm(p) sqrt(dot(p,p))
// coord macro
#define dnorm(p,q) ((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y))

typedef unsigned char uchar;

#define TEX_NUM 7
texture<uchar4, 2, cudaReadModeNormalizedFloat> texRef0;
texture<uchar,  2, cudaReadModeNormalizedFloat> texRef1;
texture<float4, 2, cudaReadModeElementType> texRef2;
texture<float4, 2, cudaReadModeElementType> texRef3;
texture<float,  2, cudaReadModeElementType> texRef4;
texture<float,  2, cudaReadModeElementType> texRef5;
texture<float2, 2, cudaReadModeElementType> texRef6;
cudaArray* cuArray[TEX_NUM];

#define i(x,y)  tex2D((texRef0),x,y)
#define t(x,y)  tex2D((texRef1),x,y)
#define f(x,y)  tex2D((texRef2),x,y)
#define b(x,y)  tex2D((texRef3),x,y)
#define da(x,y) tex2D((texRef4),x,y)
#define a(x,y)  tex2D((texRef5),x,y)
#define n(x,y)  tex2D((texRef6),x,y)

#define kernel \
    int by = blockIdx.y*BLOCK_LEN;\
    int x = blockIdx.x*BLOCK_SIZE+threadIdx.x;\
    if (x >= w) return;\
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

#define INFINITY 1E+37

__global__
void preflood(float4 *image, float2 *nearest, float key, int w, int h)
{
    kernel {
        if (a(x,y) == key) {
            nearest[y*w+x].x = x;
            nearest[y*w+x].y = y;
            image[y*w+x] = i(x,y);
        } else {
            nearest[y*w+x].x = INFINITY;
            nearest[y*w+x].y = INFINITY;
            image[y*w+x] = float4();
        }
    }
}
__global__
void flood(float2 *nearest, int step, int w, int h)
{
    kernel {
        float2 position = {x, y};
        float2 curNearest;
        float  curDistance = INFINITY;
        for (int i=x-step; i<=x+step; i+=step) {
            for (int j=y-step; j<=y+step; j+=step) {
                float2 newNearest = n(i, j);
                float  newDistance = dnorm(newNearest, position);
                if (newDistance < curDistance) {
                    curDistance = newDistance;
                    curNearest = newNearest;
                }
            }
        }
        nearest[y*w+x] = curNearest;
    }
}

__global__
void postflood(float4 *image, float key, int w, int h)
{
    kernel {
        float2 curNearest = n(x,y);
        image[y*w+x] = i(curNearest.x, curNearest.y);
    }
}

__global__
void alphaFromTrimap(float *data, int w, int h)
{
    kernel {
        data[y*w+x] = t(x,y);
    }
}

__global__
void alphaGradient(float *data, int w, int h)
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

__global__
void alphaInitialize(float *data, int w, int h)
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

__global__
void alphaReconstruct(float *data, int w, int h)
{
    kernel {
        if (a(x,y) != 0.0f && a(x,y) != 1.0f) {
            data[y*w+x] = 0.25f*(a(x-1,y)+a(x+1,y)+a(x,y-1)+a(x,y+1)-da(x,y));
        }
    }
}

__global__
void alphaRefinement(float *data, int w, int h)
{
    kernel {
        float alpha = data[y*w+x];
        alpha = (alpha-0.05f)/0.95f;
        alpha = min(max(alpha, 0.0f), 1.0f);
        data[y*w+x] = alpha;
    }
}

__global__
void alphaOutput(float4 *data, int w, int h)
{
    kernel {
        float alpha = a(x,y);
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

    cudaMallocArray(&cuArray[0], &channelDescB4, w, h);
    cudaMallocArray(&cuArray[1], &channelDescB, w, h);
    cudaMallocArray(&cuArray[2], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[3], &channelDescF4, w, h);
    cudaMallocArray(&cuArray[4], &channelDescF, w, h);
    cudaMallocArray(&cuArray[5], &channelDescF, w, h);
    cudaMallocArray(&cuArray[6], &channelDescF2, w, h);

    cudaBindTextureToArray(texRef0, cuArray[0]);
    cudaBindTextureToArray(texRef1, cuArray[1]);
    cudaBindTextureToArray(texRef2, cuArray[2]);
    cudaBindTextureToArray(texRef3, cuArray[3]);
    cudaBindTextureToArray(texRef4, cuArray[4]);
    cudaBindTextureToArray(texRef5, cuArray[5]);
    cudaBindTextureToArray(texRef6, cuArray[6]);

    cudaMalloc((void**)&alpha, w*h*sizeof(float));
    cudaMalloc((void**)&nearest, w*h*sizeof(float2));
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
    cudaUnbindTexture(texRef6);

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

    // texture 0: image
    cudaMemcpyToArrayAsync(cuArray[0], 0, 0, h_src, w*h*sizeof(uchar4), cudaMemcpyHostToDevice, 0);

    // texture 1: trimap
    cudaMemcpyToArrayAsync(cuArray[1], 0, 0, h_trimap, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice, 0);

    // texture 4: alpha
    alphaFromTrimap <<< grid, threads, 0, 0 >>> (alpha, w, h);
    cudaMemcpyToArrayAsync(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice, 0);

    for (int time=0; time<4; time++) {

    /*
     *  Calculate Foreground
     */

    // texture 2: forground image
    preflood  <<< grid, threads, 0, 0 >>> (image, nearest, 1.0f, w, h);
    cudaMemcpyToArrayAsync(cuArray[2], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice, 0);
    for (int step=maxJump; step>0; step/=2) {
        cudaMemcpyToArrayAsync(cuArray[6], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice, 0);
        flood <<< grid, threads, 0, 0 >>> (nearest, step, w, h);
    }
    cudaMemcpyToArrayAsync(cuArray[6], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice, 0);
    postflood <<< grid, threads, 0, 0 >>> (image, 1.0f, w, h);

    cudaMemcpyToArrayAsync(cuArray[2], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice, 0);

    /*
     *  Calculate Background
     */

    // texture 3: background image
    preflood  <<< grid, threads, 0, 0 >>> (image, nearest, 0.0f, w, h);
    cudaMemcpyToArrayAsync(cuArray[3], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice, 0);
    for (int step=6; step>0; step/=2) {
        cudaMemcpyToArrayAsync(cuArray[6], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice, 0);
        flood <<< grid, threads, 0, 0 >>> (nearest, step, w, h);
    }
    cudaMemcpyToArrayAsync(cuArray[6], 0, 0, nearest, w*h*sizeof(float2), cudaMemcpyDeviceToDevice, 0);
    postflood <<< grid, threads, 0, 0 >>> (image, 0.0f, w, h);
    cudaMemcpyToArrayAsync(cuArray[3], 0, 0, image, w*h*sizeof(float4), cudaMemcpyDeviceToDevice, 0);

    /*
     *  Calculate Alpha
     */

    // texture 4: gradient alpha
    alphaGradient <<< grid, threads, 0, 0 >>> (alpha, w, h);
    cudaMemcpyToArrayAsync(cuArray[4], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice, 0);
    alphaInitialize <<< grid, threads, 0, 0 >>> (alpha, w, h);
    for (int i=0; i<32; i++){
        // texture 5: alpha
        cudaMemcpyToArrayAsync(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice, 0);
        alphaReconstruct <<< grid, threads, 0, 0 >>> (alpha, w, h);
    }
    alphaRefinement <<< grid, threads, 0, 0 >>> (alpha, w, h);
    cudaMemcpyToArrayAsync(cuArray[5], 0, 0, alpha, w*h*sizeof(float), cudaMemcpyDeviceToDevice, 0);

    }

    // output alpha
    alphaOutput <<< grid, threads, 0, 0 >>> ((float4*)d_dst, w, h);
}
