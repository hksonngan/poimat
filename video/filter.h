#ifndef FILTER_H
#define FILTER_H

extern "C" void initializeTexture(
                unsigned int w, unsigned int h);

extern "C" void releaseTexture();

extern "C" void poissonFilter(
                const unsigned char *src, const unsigned char *trimap, float *dst,
                unsigned int w, unsigned int h);

#endif // FILTER_H
