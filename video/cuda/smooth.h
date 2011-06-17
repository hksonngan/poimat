#ifndef SMOOTH_H
#define SMOOTH_H

void bindSmoothTexture(cudaArray* arrImage, cudaArray* arrTemp, cudaArray* arrNearset);
void unbindSmoothTexture();

__global__ void smoothScan(float4 *image, int orientation, int w, int h);
__global__ void smoothAverage(float4 *image, int step, int w, int h);

#endif // SMOOTH_H
