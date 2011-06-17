#ifndef FILTER_H
#define FILTER_H

void initializeFilter(unsigned int w, unsigned int h);

void releaseFilter();

void poissonFilter(const unsigned char *src, const unsigned char *trimap, float *dst);

void trimapFilter(const unsigned char *src, unsigned char *trimap, int* f_point, int f_count, int* b_point, int b_count);

#endif // FILTER_H
