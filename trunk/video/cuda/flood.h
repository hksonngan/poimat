#ifndef FLOOD_H
#define FLOOD_H

void bindFloodTexture(cudaArray* arrImage, cudaArray* arrNearset, cudaArray* arrAlpha);
void unbindFloodTexture();

__global__ void floodInitialize(float4 *image, float2 *nearest, float key, int w, int h);
__global__ void floodStep(float2 *nearest, float key, int step, int w, int h);
__global__ void floodFinalize(float4 *image, float key, int radius, int w, int h);

#endif // FLOOD_H
