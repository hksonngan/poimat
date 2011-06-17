#ifndef KERNEL_H
#define KERNEL_H

#define BLOCK_SIZE   128
#define BLOCK_LEN    128
#define COLOR        4
#define INFINITY     __int_as_float(0x7f800000)
// int macro
#define div(n,m) (((n)+(m-1))/(m))
// color macro
#define dotC(p,q) (2.0f*(p.x*q.x)+4.0f*(p.y*q.y)+3.0f*(p.z*q.z))
#define normC(p,q) sqrt(2.0f*(p.x-q.x)*(p.x-q.x)+4.0f*(p.y-q.y)*(p.y-q.y)+3.0f*(p.z-q.z)*(p.z-q.z))
#define luma(p)   (0.299f*p.x+0.587f*p.y+0.114f*p.z)
// coord macro
#define norm2D(p,q) ((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y))

typedef unsigned char uchar;

// color space YUV
#define Ybits 8
#define Ubits 16
#define Vbits 16
#define COLOR_SIZE (Ybits*Ubits*Vbits)
// temperal space
#define Tbits 3
#define TEMP_SIZE (1<<Tbits)
#define HIST_SIZE (COLOR_SIZE+TEMP_SIZE)

#define kernel \
    int by = blockIdx.y*BLOCK_LEN;\
    int bx = blockIdx.x*BLOCK_SIZE;\
    int x = bx+threadIdx.x;\
    if (x >= w) return;\
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

#define loop \
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

////////////////////////////////////
// operator for int2

__device__ inline
int2 operator +(int2 &op1, int2 &op2)
{
    int2 result = {
        op1.x + op2.x,
        op1.y + op2.y
    };
    return result;
}

__device__ inline
int2 operator +=(int2 &op1, int2 &op2)
{
    op1.x += op2.x;
    op1.y += op2.y;
    return op1;
}

__device__ inline
int2 operator /(int2 op1, float op2)
{
    int2 result = {
        op1.x / op2,
        op1.y / op2
    };
    return result;
}

__device__ inline
int2 operator *(int2 op1, float op2)
{
    int2 result = {
        op1.x * op2,
        op1.y * op2
    };
    return result;
}

__device__ inline
int2 operator *=(int2 op1, float op2)
{
    op1.x *= op2;
    op1.y *= op2;
    return op1;
}


////////////////////////////////////
// operator for float4

__device__ inline
float4 operator +(float4 &op1, float4 &op2)
{
    float4 result = {
        op1.x + op2.x,
        op1.y + op2.y,
        op1.z + op2.z,
        op1.w + op2.w
    };
    return result;
}

__device__ inline
float4 operator -(float4 &op1, float4 &op2)
{
    float4 result = {
        op1.x - op2.x,
        op1.y - op2.y,
        op1.z - op2.z,
        op1.w - op2.w
    };
    return result;
}

__device__ inline
float4 operator *(float op1, float4 op2)
{
    float4 result = {
        op1 * op2.x,
        op1 * op2.y,
        op1 * op2.z,
        op1 * op2.w
    };
    return result;
}

__device__ inline
float4 operator /(float4 op1, float op2)
{
    float4 result = {
        op1.x / op2,
        op1.y / op2,
        op1.z / op2,
        op1.w / op2
    };
    return result;
}

__device__ inline
float4 operator +=(float4 &op1, float4 &op2)
{
    op1.x += op2.x;
    op1.y += op2.y;
    op1.z += op2.z;
    op1.w += op2.w;
    return op1;
}

__device__ inline
float4 operator -=(float4 &op1, float4 &op2)
{
    op1.x -= op2.x;
    op1.y -= op2.y;
    op1.z -= op2.z;
    op1.w -= op2.w;
    return op1;
}

__device__ inline
float4 operator /=(float4 &op1, float &op2)
{
    op1.x /= op2;
    op1.y /= op2;
    op1.z /= op2;
    op1.w /= op2;
    return op1;
}

__device__ inline
float4 operator *=(float4 &op1, float &op2)
{
    op1.x *= op2;
    op1.y *= op2;
    op1.z *= op2;
    op1.w *= op2;
    return op1;
}

#endif // KERNEL_H
