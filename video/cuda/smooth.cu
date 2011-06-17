#include "kernel.h"

texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<float4, 2, cudaReadModeElementType>     texTemp;
texture<float2, 2, cudaReadModeElementType>     texNearest;

#define i(x,y)  tex2D(texImage,x,y)
#define p(x,y)  tex2D(texTemp,x,y)
#define n(x,y)  tex2D(texNearest,x,y)

void bindSmoothTexture(cudaArray* arrImage, cudaArray* arrTemp, cudaArray* arrNearset)
{
    cudaBindTextureToArray(texImage,   arrImage);
    cudaBindTextureToArray(texTemp,    arrTemp);
    cudaBindTextureToArray(texNearest, arrNearset);
}

void unbindSmoothTexture()
{
    cudaUnbindTexture(texImage);
    cudaUnbindTexture(texTemp);
    cudaUnbindTexture(texNearest);
}

__global__
void smoothScan(float4 *image, int orientation, int w, int h)
{
    __shared__ float4 smem[BLOCK_SIZE*2];

    int tid = threadIdx.x;
    int offset = 1;

    int px, py, qx, qy;
    if (orientation) {
        px = blockIdx.x*BLOCK_SIZE*2+tid*2+0;
        py = blockIdx.y;
        qx = blockIdx.x*BLOCK_SIZE*2+tid*2+1;
        qy = blockIdx.y;
    } else {
        px = blockIdx.y;
        py = blockIdx.x*BLOCK_SIZE*2+tid*2+0;
        qx = blockIdx.y;
        qy = blockIdx.x*BLOCK_SIZE*2+tid*2+1;
    }

    smem[tid*2+0] = p(px,py);
    smem[tid*2+1] = p(qx,qy);

    for (int d=BLOCK_SIZE; d>0; d/=2) {
        __syncthreads();
        if (tid < d) {
            int ai = offset*(tid*2+1)-1;
            int bi = offset*(tid*2+2)-1;
            smem[bi] += smem[ai];
        }
        offset *= 2;
    }

    if (tid == 0) {
        smem[BLOCK_SIZE*2-1] = float4();
    }

    for (int d=1; d<=BLOCK_SIZE; d*=2) {
        offset /= 2;
        __syncthreads();
        if (tid < d) {
            int ai = offset*(tid*2+1)-1;
            int bi = offset*(tid*2+2)-1;
            float4 tmem = smem[ai];
            smem[ai]  = smem[bi];
            smem[bi] += tmem;
        }
    }

    {
        __syncthreads();
        smem[tid*2+0] += p(px,py);
        smem[tid*2+1] += p(qx,qy);
        if (px < w && py < h) {
            image[py*w+px] = smem[tid*2+0];
        }
        if (qx < w && qy < h)
            image[qy*w+qx] = smem[tid*2+1];
    }
}

__global__
void smoothAverage(float4 *image, int step, int w, int h)
{
	int block = BLOCK_SIZE*2;
    kernel {
        float2 nearest = n(x,y);
        float2 position = {x,y};

        float radius = norm2D(nearest, position);
		if (radius == INFINITY) {
			image[y*w+x] = float4();
			continue;
		}
		if (radius < step*step) {
			image[y*w+x] = i(nearest.x, nearest.y);
			image[y*w+x].w = 1.0f;
			continue;
		}

        int left   = max(x-step-1, -1);
        int right  = min(x+step, w-1);
        int top    = max(y-step-1, -1);
        int bottom = min(y+step, h-1);

        float4 mixture = float4();

        mixture += p(right,bottom);
        if(left > -1)
			mixture -= p(left,bottom);
        if(top > -1)
			mixture -= p(right,top);
        if(left > -1 && top > -1)
			mixture += p(left,top);
		
		if(right/block != left/block) {
			int center = right/block*block-1;
			mixture += p(center,bottom);
			if (top > -1)
				mixture -= p(center,top);
		}
		
		if(bottom/block != top/block) {
			int middle = bottom/block*block-1;
			mixture += p(right,middle);
			if (left > -1)
				mixture -= p(left,middle);
		}

		if (mixture.w != 0.0f) {
			mixture /= mixture.w;
			mixture.w = 1.0f;
			image[y*w+x] = mixture;
		}
    }
}
