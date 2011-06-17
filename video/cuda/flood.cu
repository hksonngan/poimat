#include "kernel.h"

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<float,  2, cudaReadModeElementType>     texAlpha;
texture<float2, 2, cudaReadModeElementType>     texNearest;

#define i(x,y)  tex2D(texImage,x,y)
#define a(x,y)  tex2D(texAlpha,x,y)
#define n(x,y)  tex2D(texNearest,x,y)

void bindFloodTexture(cudaArray* arrImage, cudaArray* arrNearset, cudaArray* arrAlpha)
{
    cudaBindTextureToArray(texImage,   arrImage);
    cudaBindTextureToArray(texNearest, arrNearset);
    cudaBindTextureToArray(texAlpha,   arrAlpha);
}

void unbindFloodTexture()
{
    cudaUnbindTexture(texImage);
    cudaUnbindTexture(texNearest);
    cudaUnbindTexture(texAlpha);
}

__global__
void floodInitialize(float4 *image, float2 *nearest, float key, int w, int h)
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
void floodStep(float2 *nearest, float key, int step, int w, int h)
{ 
    kernel {
		float alpha = a(x,y);
		if (alpha == key || alpha == 1.0f-key)
            continue;
        float2 position = {x,y};
        float2 curNearest = n(x,y);
        float curDistance = norm2D(curNearest, position);
        #pragma unroll
        for (int i=-1; i<=1; i++) {
            int u = x+i*step;
            if (u < 0 || u >= w)
                continue;
            #pragma unroll
            for (int j=-1; j<=1; j+=2-i*i) {
                int v = y+j*step;
                if (v < 0 || v >= h)
                    continue;
                float2 newNearest = n(u,v);
                float newDistance = norm2D(newNearest, position);
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
void floodFinalize(float4 *image, float key, int radius, int w, int h)
{
    kernel {
		float alpha = a(x,y);
		if (alpha == key || alpha == 1.0f-key)
            continue;
        float2 position = {x,y};
        float2 curNearest = n(x,y);
        if (norm2D(position, curNearest) > radius*radius)
            continue;
        image[y*w+x] = i(curNearest.x,curNearest.y);
        image[y*w+x].w = 1.0f;
    }
}
