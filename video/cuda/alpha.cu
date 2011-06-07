#include "kernel.h"

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<uchar,  2, cudaReadModeNormalizedFloat> texTrimap;
texture<float4, 2, cudaReadModeElementType> texForeground;
texture<float4, 2, cudaReadModeElementType> texBackground;
texture<float,  2, cudaReadModeElementType> texLapAlpha;
texture<float,  2, cudaReadModeElementType> texAlpha;

#define i(x,y)  tex2D((texImage),x,y)
#define t(x,y)  tex2D((texTrimap),x,y)
#define f(x,y)  tex2D((texForeground),x,y)
#define b(x,y)  tex2D((texBackground),x,y)
#define la(x,y) tex2D((texLapAlpha),x,y)
#define a(x,y)  tex2D((texAlpha),x,y)

void bindAlphaTexture(
    cudaArray* arrImage,
    cudaArray* arrTrimap,
    cudaArray* arrForeground,
    cudaArray* arrBackground,
    cudaArray* arrLapAlpha,
    cudaArray* arrAlpha)
{
    cudaBindTextureToArray(texImage,      arrImage);
    cudaBindTextureToArray(texTrimap,     arrTrimap);
    cudaBindTextureToArray(texForeground, arrForeground);
    cudaBindTextureToArray(texBackground, arrBackground);
    cudaBindTextureToArray(texLapAlpha,   arrLapAlpha);
    cudaBindTextureToArray(texAlpha,      arrAlpha);
}

void unbindAlphaTexture()
{
    cudaUnbindTexture(texImage);
    cudaUnbindTexture(texTrimap);
    cudaUnbindTexture(texForeground);
    cudaUnbindTexture(texBackground);
    cudaUnbindTexture(texLapAlpha);
    cudaUnbindTexture(texAlpha);
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
            data[y*w+x] = 0.25f*(a(x-1,y)+a(x+1,y)+a(x,y-1)+a(x,y+1)-la(x,y));
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
