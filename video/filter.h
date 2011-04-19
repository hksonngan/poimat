#ifndef FILTER_H
#define FILTER_H

extern "C" void medianFilter(
                const unsigned char *src, unsigned char *dst,
                unsigned int w, unsigned int h,
                unsigned int r);

#endif // FILTER_H
