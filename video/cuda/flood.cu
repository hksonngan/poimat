#include "kernel.h"

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<float,  2, cudaReadModeElementType>     texAlpha;
texture<float2, 2, cudaReadModeElementType>     texNearest;

#define i(x,y)  tex2D((texImage),x,y)
#define a(x,y)  tex2D((texAlpha),x,y)
#define n(x,y)  tex2D((texNearest),x,y)

void bindFloodTexture(cudaArray* arrImage, cudaArray* arrAlpha, cudaArray* arrNearset)
{
    cudaBindTextureToArray(texImage,   arrImage);
    cudaBindTextureToArray(texAlpha,   arrAlpha);
    cudaBindTextureToArray(texNearest, arrNearset);
}

void unbindFloodTexture()
{
    cudaUnbindTexture(texImage);
    cudaUnbindTexture(texAlpha);
    cudaUnbindTexture(texNearest);
}

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
void postflood(float4 *image, int w, int h)
{
    kernel {
        float2 curNearest = n(x,y);
        image[y*w+x] = i(curNearest.x, curNearest.y);
    }
}
