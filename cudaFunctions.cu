#include "cudaFunctions.hpp"

__global__ void saveGrayscaleImage(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned char green = 128;
    unsigned char blue = 128;

    int inputIndex, outputIndex;
    for(long x=xCoord; x < xSize; x+=(blockDim.x * gridDim.x))
        {
        	for(long y=yCoord; y < ySize; y+=(blockDim.y * gridDim.y))
            {
                inputIndex = xSize * y + x;
    			outputIndex = xSize * y * 3 + x * 3;

    			imageOut[outputIndex + 2] = imageIn[inputIndex];
    			imageOut[outputIndex + 1] = green;
    			imageOut[outputIndex] = blue;
    	    }
        }
}

__global__ void makeGrayscaleImage(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize, int alpha)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    
    double redWeigth = 0.33;
    double greenWeigth = 0.34;
    double blueWeigth = 0.33;

    double currentColor;
    int inputIndex;
    for(long x=xCoord; x < xSize; x+=(blockDim.x * gridDim.x))
    {
    	for(long y=yCoord; y < ySize; y+=(blockDim.y * gridDim.y))
        {
            currentColor = 0.0;
			inputIndex = xSize * y * 3 + x * 3;

			currentColor += imageIn[inputIndex + 2] * redWeigth;
			currentColor += imageIn[inputIndex + 1] * greenWeigth;
			currentColor += imageIn[inputIndex] * blueWeigth;
			currentColor = currentColor * alpha;
			if(currentColor > 255.0) currentColor = 255;
            imageOut[xSize * y + x] = (unsigned char)currentColor;
	    }
    }
}

__global__ void makeDifferenceImage(unsigned char* grayscaleImage, unsigned char* differenceImage, long xSize, long ySize)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;
    
    if(xCoord == 0 or xCoord == xSize-1)
    {
        differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
        xCoord += xAdd;
    }
    if(yCoord == 0 or xCoord == xSize-1)
    {
        differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
        yCoord += yAdd;
    }

    unsigned char brightestPoint;
    unsigned char darkestPoint;
    for(long x=xCoord; x < xSize; x+=xAdd)
    {   
        for(long y=yCoord; y < ySize; y+=yAdd)
        {
            if(y == 0 or x == xSize-1)
            {
                differenceImage[xSize * y + x] = grayscaleImage[xSize * y + x];
                break;
            }

            brightestPoint = 0;
            darkestPoint = 255;
            for(int i=-1; i < 2; i++)
            {
                for(int j=-1; j < 2; j++)
                {
                    if(i == 0 and j == 0) continue;
                    int currentIndex = xSize * (y + j) + (x + i);
                    brightestPoint = (brightestPoint < grayscaleImage[currentIndex])?grayscaleImage[currentIndex]:brightestPoint;
                    darkestPoint = (darkestPoint > grayscaleImage[currentIndex])?grayscaleImage[currentIndex]:darkestPoint;
                }             
            }
            differenceImage[xSize * y + x] = brightestPoint - darkestPoint;
	    }
    }
}

__device__ bool checkIfHasFloodedNeighbour(bool* isPixelFlooded, long xSize, long ySize, long xCoord, long yCoord)
{
	int xStart=-1;
	int xEnd=1;
	int yStart=-1;
	int yEnd=1;

	if(xCoord == 0) xStart = 0;
	if(xCoord == xSize-1) xEnd = 0;

	if(yCoord == 0) yStart = 0;
	if(yCoord == ySize-1) yEnd = 0;

	for(int x=xStart; x <= xEnd; x++)
	{
		for(int y=yStart; y <= yEnd; y++)
		{
			if(isPixelFlooded[xSize * (yCoord + y) + (xCoord + x)]) return true;
		}
	}
	return false;
}

__global__ void makeWatershade(unsigned char* differenceImage, bool* isPixelFlooded, long xSize, long ySize, bool* hasChangedAnyPixel, unsigned char treshold)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;

    for(long x=xCoord; x < xSize; x+=xAdd)
    {
        for(long y=yCoord; y < ySize; y+=yAdd)
        {
        	if((not isPixelFlooded[xSize*y+x]) and (differenceImage[xSize*y+x] <= treshold))
        	{
        		if(checkIfHasFloodedNeighbour(isPixelFlooded, xSize, ySize, x, y))
        		{
        			isPixelFlooded[xSize*y+x] = true;
        			*hasChangedAnyPixel = true;
        		}
        	}
	    }
    }

}

__global__ void brightenImage(unsigned char* image, long xSize, long ySize, double alpha)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;

    for(long x=xCoord; x < xSize; x+=xAdd)
    {
        for(long y=yCoord; y < ySize; y+=yAdd)
        {
        	image[xSize*y+x] = image[xSize*y+x] * alpha;
	    }
    }
}

__global__ void makeGaussianBlur(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize)
{
    int gaussianMask[5][5] = {{1, 1, 2, 1, 1},
                              {1, 2, 4, 2, 1},
                              {2, 4, 8, 4, 2},
                              {1, 2, 4, 2, 1},
                              {1, 1, 2, 1, 1}};
    int maskSum = 52;

    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;

    while(xCoord < 2)
    {
        xCoord += xAdd;
    }
    while(yCoord < 2)
    {
        yCoord += yAdd;
    }
    
    long inputIndex, outputIndex;
    int currentColor;
    double divide;
    //blur grayscale image
    for(long x=xCoord; x < xSize-2; x+=xAdd)
    {
        for(long y=yCoord; y < ySize-2; y+=yAdd)
        {
            currentColor = 0;
            //blur single pixel
            for (int i = 0; i<5; i++)
            {
		        for (int j = 0; j<5; j++)
                {
			        inputIndex = xSize*(y + i - 2) + (x + j - 2);
			        currentColor += imageIn[inputIndex] * gaussianMask[i][j];
		        }
	        }
            outputIndex = xSize * y + x;
            divide = currentColor / maskSum;
            if(divide > 255.0) divide = 255;
            imageOut[outputIndex] = divide;
        }
    }
}

__global__ void markFloodedPixels(unsigned char* imageIn, bool* isFlooded, long xSize, long ySize)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;

    unsigned char red = 0;
    unsigned char green = 166;
    unsigned char blue = 147;

    int pixelIndex;
    for(long x=xCoord; x < xSize; x+=xAdd)
    {
    	yCoord = blockIdx.y * blockDim.y + threadIdx.y;
        for(long y=yCoord; y < ySize; y+=yAdd)
        {
            if(isFlooded[xSize * y + x])
            {
            	pixelIndex = xSize * y * 3 + x * 3;
            	imageIn[pixelIndex + 2] = red;
            	imageIn[pixelIndex + 1] = green;
            	imageIn[pixelIndex] = blue;
            }
	    }
    }
}

__global__ void markStartPixel(bool* isFlooded, long xSize, long x, long y)
{
	isFlooded[xSize*y+x] = true;
}