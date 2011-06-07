#ifndef ALPHA_H
#define ALPHA_H

void bindAlphaTexture(cudaArray* arrImage, cudaArray* arrTrimap,
                      cudaArray* arrForeground, cudaArray* arrBackground,
                      cudaArray* arrLapAlpha, cudaArray* arrAlpha);
void unbindAlphaTexture();

__global__ void alphaFromTrimap(float *data, int w, int h);
__global__ void alphaGradient(float *data, int w, int h);
__global__ void alphaInitialize(float *data, int w, int h);
__global__ void alphaReconstruct(float *data, int w, int h);
__global__ void alphaRefinement(float *data, int w, int h);
__global__ void alphaOutput(float4 *data, int w, int h);

#endif // ALPHA_H
