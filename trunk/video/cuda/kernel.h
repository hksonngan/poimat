#ifndef KERNEL_H
#define KERNEL_H

#define BLOCK_SIZE   128
#define BLOCK_LEN    128
#define COLOR        4
#define INFINITY     1E+37

// int macro
#define div(n,m) (((n)+(m-1))/(m))
// color macro
#define dot(p,q) (2.0f*(p.x*q.x)+4.0f*(p.y*q.y)+3.0f*(p.z*q.z))
#define norm(p) sqrt(dot(p,p))
// coord macro
#define dnorm(p,q) ((p.x-q.x)*(p.x-q.x)+(p.y-q.y)*(p.y-q.y))

typedef unsigned char uchar;

#define kernel \
    int by = blockIdx.y*BLOCK_LEN;\
    int x = blockIdx.x*BLOCK_SIZE+threadIdx.x;\
    if (x >= w) return;\
    for (int y=by; y<min(by+BLOCK_LEN, h); y++)

#endif // KERNEL_H
