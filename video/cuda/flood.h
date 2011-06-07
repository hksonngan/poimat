#ifndef FLOOD_H
#define FLOOD_H

void bindFloodTexture(cudaArray* arrImage, cudaArray* arrAlpha, cudaArray* arrNearset);
void unbindFloodTexture();

__global__ void preflood(float4 *image, float2 *nearest, float key, int w, int h);
__global__ void flood(float2 *nearest, int step, int w, int h);
__global__ void postflood(float4 *image, int w, int h);

#endif // FLOOD_H
