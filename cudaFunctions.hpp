#pragma once

#include "cuda_runtime.h"

__global__ void saveGrayscaleImage(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize);

__global__ void makeGrayscaleImage(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize, int alpha);

__global__ void makeDifferenceImage(unsigned char* grayscaleImage, unsigned char* differenceImage, long xSize, long ySize);

__device__ bool checkIfHasFloodedNeighbour(bool* isPixelFlooded, long xSize, long ySize, long xCoord, long yCoord);

__global__ void makeWatershade(unsigned char* differenceImage, bool* isPixelFlooded, long xSize, long ySize, bool* hasChangedAnyPixel, unsigned char treshold);

__global__ void brightenImage(unsigned char* image, long xSize, long ySize, double alpha);

__global__ void makeGaussianBlur(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize);

__global__ void markFloodedPixels(unsigned char* imageIn, bool* isFlooded, long xSize, long ySize);

__global__ void markStartPixel(bool* isFlooded, long xSize, long x, long y);