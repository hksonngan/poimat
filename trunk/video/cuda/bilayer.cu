#include "kernel.h"
#include "bilayer.h"

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<uchar,  2, cudaReadModeElementType>     texTemperal;
texture<int2,   2, cudaReadModeElementType>     texHistogram;
texture<float2, 2, cudaReadModeElementType>     texNearest;
texture<float4, 2, cudaReadModeElementType>     texGraph;
texture<float,  2, cudaReadModeElementType>     texAlpha;

#define i(x,y)  tex2D(texImage,x,y)
#define t(x,y)  tex2D(texTemperal,x,y)
#define h(i)    tex2D(texHistogram,i,1)
#define n(x,y)  tex2D(texNearest,x,y)
#define g(x,y)  tex2D(texGraph,x,y)
#define a(x,y)  tex2D(texAlpha,x,y)

void bindBilayerTexture(cudaArray* arrImage, cudaArray* arrTemperal, cudaArray* arrHistogram, cudaArray* arrNearest, cudaArray* arrGraph, cudaArray* arrAlpha)
{
    cudaBindTextureToArray(texImage,     arrImage);
    cudaBindTextureToArray(texTemperal,  arrTemperal);
    cudaBindTextureToArray(texHistogram, arrHistogram);
    cudaBindTextureToArray(texNearest,   arrNearest);
    cudaBindTextureToArray(texGraph,     arrGraph);
    cudaBindTextureToArray(texAlpha,     arrAlpha);
}

void unbindBilayerTexture()
{
    cudaUnbindTexture(texImage);
    cudaUnbindTexture(texTemperal);
    cudaUnbindTexture(texHistogram);
    cudaUnbindTexture(texNearest);
    cudaUnbindTexture(texGraph);
    cudaUnbindTexture(texAlpha);
}

__device__ inline
int colorIndex(float4 color)
{
    float Y = 0.299f*color.x+0.587f*color.y+0.114f*color.z;
    float U = 0.5f-0.169f*color.x-0.331f*color.y+0.5f*color.z;
    float V = 0.5f+0.5f*color.x-0.419f*color.y-0.081f*color.z;
    return ((int)(Y*Ybits))+
           ((int)(U*Ubits))*Ybits+
           ((int)(V*Vbits))*Ybits*Ubits;
}

__global__ void graphIterate(int2 *histogram)
{
    int tid = blockIdx.x*BLOCK_SIZE+threadIdx.x;
    histogram[tid] *= 0.5f;
}

__global__
void graphStatistic(int2 *histogram, uchar *temperal, int w, int h)
{
    kernel {
        int indexC = colorIndex(i(x,y));
		uchar indexT = t(x,y);
        float alpha = a(x,y);
#if __CUDA_ARCH__ >= 110
        if (alpha == 1.0f) {
            atomicAdd(&(histogram[indexC].x), 1);
            atomicAdd(&(histogram[COLOR_SIZE+indexT&(TEMP_SIZE-1)].x), 1);
			indexT = indexT<<1 | 1;
        }
        if (alpha == 0.0f) {
            atomicAdd(&(histogram[indexC].y), 1);
            atomicAdd(&(histogram[COLOR_SIZE+indexT&(TEMP_SIZE-1)].y), 1);
			indexT = indexT<<1 | 0;
        }
		temperal[y*w+x] = indexT;
#endif
    }
}

#define Frac(a,b) (a+1.0f)/(b+2.0f)

__global__
void graphConstruct(float4 *graph, int w, int h)
{
    kernel {
		// spatial
		float4 pixel = i(x,y);
		float4 pixelX = i(x+1,y);
		float4 pixelY = i(x,y+1);
		// temperal
		char indexT = t(x,y);
		float2 ratioT = {h(COLOR_SIZE+indexT).x, h(COLOR_SIZE+indexT).y};
		// color
        int indexC = colorIndex(pixel);
        float2 ratioC = {h(indexC).x, h(indexC).y};
		// calculation
        float4 node;
        node.x = __expf(-normC(pixel, pixelX)/5.0f);
        node.y = __expf(-normC(pixel, pixelY)/5.0f);
        node.z = -__logf(Frac(ratioC.x, ratioC.x+ratioC.y)*Frac(ratioT.x, ratioT.x+ratioT.y));
        node.w = -__logf(Frac(ratioC.y, ratioC.x+ratioC.y)*Frac(ratioT.y, ratioT.x+ratioT.y));
        graph[y*w+x] = node;
    }
}

__global__
void graphCut(float *alpha, int w, int h)
{
    kernel {
		float4 node = g(x,y);
        float energy = node.z-node.w;
        if (x > 0)   energy += (a(x-1,y) > 0.5f ? -1.0f : 1.0f)*g(x-1,y).x;
        if (x < w-1) energy += (a(x+1,y) > 0.5f ? -1.0f : 1.0f)*node.x;
        if (y > 0)   energy += (a(x,y-1) > 0.5f ? -1.0f : 1.0f)*g(x,y-1).y;
        if (y < h-1) energy += (a(x,y+1) > 0.5f ? -1.0f : 1.0f)*node.y;
        alpha[y*w+x] = energy > 0.0f ? 0.0f : 1.0f;
    }
}

__global__
void trimapInitialize(float2 *neaerest, int w, int h)
{
    kernel {
        float value = a(x,y);
        float2 curNearest = {INFINITY, INFINITY};
        if ((x > 0   && a(x-1,y) != value) ||
            (x < w-1 && a(x+1,y) != value) ||
            (y > 0   && a(x,y-1) != value) ||
            (y < h-1 && a(x,y+1) != value)) {
            curNearest.x = x;
            curNearest.y = y;
        }
        neaerest[y*w+x] = curNearest;
    }
}

__global__
void trimapPoint (int2 *points, int count, float value, float *alpha, float2 *nearest, int w, int h)
{
	int tid = blockIdx.x*BLOCK_SIZE+threadIdx.x;
	if (tid >= count)
		return;
	int2   point    = points[tid];
	float2 position = {point.x, point.y};
	alpha  [point.y*w+point.x] = value;
	nearest[point.y*w+point.x] = position;
}

__global__
void trimapOutput(uchar *trimap, int radius, int w, int h)
{
    kernel {
        float2 position = {x,y};
        float2 curNearest = n(x,y);
        unsigned char value = 128;
        if (norm2D(position, curNearest) > radius*radius) {
            value = a(x,y) > 0.5f ? 255 : 0;
        }
        trimap[y*w+x] = value;
    }
}
