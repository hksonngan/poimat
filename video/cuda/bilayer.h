#ifndef BILAYER_H
#define BILAYER_H
#include "kernel.h"

void bindBilayerTexture(cudaArray* arrImage, cudaArray* arrTemperal, cudaArray* arrHistogram, cudaArray* arrNearest, cudaArray* arrGraph, cudaArray* arrAlpha);
void unbindBilayerTexture();

__global__ void graphStatistic(int2 *histogram, uchar *temperal, int w, int h);
__global__ void graphIterate(int2 *histogram);
__global__ void graphConstruct(float4 *graph, int w, int h);
__global__ void graphCut(float *alpha, int w, int h);
__global__ void trimapInitialize(float2 *neaerest, int w, int h);
__global__ void trimapPoint (int2 *points, int count, float value, float *alpha, float2 *nearest, int w, int h);
__global__ void trimapOutput(uchar *trimap, int radius, int w, int h);

#endif // BILAYER_H
