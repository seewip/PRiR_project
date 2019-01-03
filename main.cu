#include "cuda_runtime.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <string> 

using namespace cv;
using namespace std;

#define HANDLE_ERROR(ans) { handleCudaError((ans), __FILE__, __LINE__); }
inline void handleCudaError(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"[ERROR]: %s in %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


struct Coords
{
    int width;
    int heigth;
};

class ImageManager
{
public:
    ImageManager() = default;
    ~ImageManager() = default;

    Coords readImage(string imagePath);  
    void saveImage(string outputPath);
    void setBlockSize(int blockSize);
    void setGridSize(int gridSize);
    int getBlockSize();
    int getGridSize();
    void setStartPixel(Coords startCoords);
    void setTreshold(int treshold);
    void clearRunningTime();
    double getRunningTime();
    void performWatershade();
private:
    void prepareOutputImage(Mat& inputImage);

    Mat inputImage;
    Mat outputImage;
    int blockSize;
    int gridSize;
    Coords startPixel;
    int treshold;
    float runningTime;
};

struct RGB
{
    int red;
    int green;
    int blue;
};

__global__ void makeGrayscaleImage(unsigned char* imageIn, unsigned char* imageOut, long xSize, long ySize)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;
    
    double redWeigth = 0.33;    
    double greenWeigth = 0.34;
    double blueWeigth = 0.33;

    double currentColor;
    int inputIndex;
    while(xCoord < xSize)
    {
        while(yCoord < ySize)
        {
            currentColor = 0.0;
			inputIndex = xSize * yCoord * 3 + xCoord * 3;

			currentColor += imageIn[inputIndex + 2] * redWeigth;
			currentColor += imageIn[inputIndex + 1] * greenWeigth;
			currentColor += imageIn[inputIndex] * blueWeigth;
            imageOut[xSize * yCoord + xCoord] = (unsigned char)currentColor;
            yCoord += yAdd;
	    }
        xCoord += xAdd;
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
    while(xCoord < xSize)
    {   
        if(xCoord == xSize-1)
        {
            differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
            break;
        }
        while(yCoord < ySize)
        {
            if(yCoord == 0 or xCoord == xSize-1)
            {
                differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
                break;
            }

            brightestPoint = 0;
            darkestPoint = 255;
            for(int x=-1; x < 2; x++)
            {
                for(int y=-1; y < 2; y++)
                {
                    if(x == 0 and y == 0)
                    {
                        continue;
                    }
                    int currentIndex = xSize * (yCoord + y) + (xCoord + x);  
                    brightestPoint = (brightestPoint < grayscaleImage[currentIndex])?grayscaleImage[currentIndex]:brightestPoint;
                    darkestPoint = (darkestPoint > grayscaleImage[currentIndex])?grayscaleImage[currentIndex]:darkestPoint;
                }             
            }
            differenceImage[xSize * yCoord + xCoord] = brightestPoint - darkestPoint;
            yCoord += yAdd;
	    }
        xCoord+= xAdd;
    }
}

__device__ bool cehckIfHasFloodedNeighbour(bool* isPixelFlooded, long xSize, long ySize, long xCoord, long yCoord)
{
	long xStart=-1;
	long xEnd=1;
	long yStart=-1;
	long yEnd=1;

	if(xCoord == 0) xStart = 0;
	else if(xCoord == xSize-1) xEnd = 0;
	if(yCoord == 0) yStart = 0;
	else if(yCoord == ySize-1) yEnd = 0;

	while(xStart <= xEnd)
	{
		while(yStart <= yEnd)
		{
			if(xStart==0 and yStart==0) continue;
			if(isPixelFlooded[xSize * (yCoord + yStart) + (xCoord + xStart)])
				return true;
			yStart++;
		}
		xStart++;
	}
	return false;
}

__global__ void makeWatershade(unsigned char* differenceImage, bool* isPixelFlooded, long xSize, long ySize, bool* hasChangedAnyPixel, int treshold)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    long xAdd = blockDim.x * gridDim.x;
    long yAdd = blockDim.y * gridDim.y;
    while(xCoord < xSize)
    {
        while(yCoord < ySize)
        {
        	if(not isPixelFlooded[xSize*yCoord+xCoord] /*and differenceImage[xSize*yCoord+xCoord] <= treshold*/)
        	{
        		if(cehckIfHasFloodedNeighbour(isPixelFlooded, xSize, ySize, xCoord, yCoord))
        		{
        			isPixelFlooded[xSize*yCoord+xCoord] = true;
        			*hasChangedAnyPixel = true;
        		}
        	}
        	yCoord+= yAdd;

	    }
        xCoord+= xAdd;
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

    while(xCoord <= 1)
    {
        xCoord += xAdd;
    }
    while(yCoord <= 1)
    {
        yCoord += yAdd;
    }
    
    long inputIndex, outputIndex;
    unsigned char currentColor;
    
    //blur grayscale image
    while(xCoord < xSize-2)
    {
        while(yCoord < ySize-2)
        {
            currentColor = 0;
            //blur single pixel
            for (int y = 0; y<5; y++)
            {
		        for (int x = 0; x<5; x++)
                {
			        inputIndex = xSize*(yCoord + y - 2) + (xCoord + x - 2);

			        currentColor += imageIn[inputIndex] * gaussianMask[x][y];
		        }
	        }
            outputIndex = xSize * yCoord + xCoord;
            imageOut[outputIndex] = currentColor / maskSum;
            yCoord+= yAdd;
        }
        xCoord+= xAdd;
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
    while(xCoord < xSize)
    {
        while(yCoord < ySize)
        {
            if(isFlooded[xSize * yCoord + xCoord])
            {
            	pixelIndex = xSize * yCoord * 3 + xCoord * 3;
            	imageIn[pixelIndex + 2] = red;
            	imageIn[pixelIndex + 1] = green;
            	imageIn[pixelIndex] = blue;
            }
            yCoord += yAdd;
	    }
        xCoord += xAdd;
    }
}

__global__ void markStartPixel(bool* isFlooded, long xSize, long x, long y)
{
	isFlooded[xSize*y+x] = true;
}


Coords ImageManager::readImage(string imagePath)
{
    inputImage = imread(imagePath, CV_LOAD_IMAGE_COLOR);
    if (not inputImage.data)
    {
	    cout << "[ERROR] Could not open input image" << endl;
	    exit(-1);
    }
    Coords readImgData{inputImage.cols, inputImage.rows};
    return readImgData;
}

void ImageManager::performWatershade()
{
	cout << "Treshold = " << treshold << endl;
	cout << "blockSize = " << blockSize << endl;
	cout << "gridSize = " << gridSize << endl;
    long cudaGrayscaleImageSize = sizeof(unsigned char) * inputImage.rows * inputImage.cols;
    long cudaInputImageSize = cudaGrayscaleImageSize * 3;
    dim3 block(blockSize, blockSize);
    dim3 grid(gridSize, gridSize);

    cudaEvent_t startTime, stopTime;

    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, 0);

    //allocate memory on GPU
    unsigned char* cudaInputImage; 
    HANDLE_ERROR(cudaMalloc((void**)& cudaInputImage, cudaInputImageSize));
    HANDLE_ERROR(cudaMemcpy(cudaInputImage, inputImage.data, cudaInputImageSize, cudaMemcpyHostToDevice));
    
    unsigned char* cudaGrayscaleImage;
    HANDLE_ERROR(cudaMalloc((void**)& cudaGrayscaleImage, cudaGrayscaleImageSize));

    makeGrayscaleImage << < grid, block >> > (cudaInputImage, cudaGrayscaleImage, inputImage.cols, inputImage.rows);
    
    cudaDeviceSynchronize();
    cout << "After grayscale" << endl;
    
    unsigned char* cudaBlurredImage;
    HANDLE_ERROR(cudaMalloc((void**)& cudaBlurredImage, cudaGrayscaleImageSize));
    
    makeGaussianBlur << < grid, block >> > (cudaGrayscaleImage, cudaBlurredImage, inputImage.cols, inputImage.rows);
    
    cudaDeviceSynchronize();
    cout << "After blur" << endl;

    HANDLE_ERROR(cudaFree(cudaGrayscaleImage));
    unsigned char* cudaDifferenceImage;
    HANDLE_ERROR(cudaMalloc((void**)& cudaDifferenceImage, cudaGrayscaleImageSize));
    
    makeDifferenceImage<< < grid, block >> >(cudaBlurredImage, cudaDifferenceImage, inputImage.cols, inputImage.rows);
    
    cudaDeviceSynchronize();
    cout << "After difference" << endl;

    HANDLE_ERROR(cudaFree(cudaBlurredImage));

    bool* isFlooded;
    long cudaBooleanImageSize = sizeof(bool) * inputImage.rows * inputImage.cols;
    HANDLE_ERROR(cudaMalloc((void**)& isFlooded, cudaBooleanImageSize));

    bool* hasAnyPixelBeenChanged;
    HANDLE_ERROR(cudaMalloc((void**)& hasAnyPixelBeenChanged, sizeof(bool)));

    bool hostHasBeenChanged;
    HANDLE_ERROR(cudaMemset(isFlooded, 0,cudaBooleanImageSize));
    markStartPixel<<<1,1>>>(isFlooded, inputImage.cols, startPixel.width, startPixel.heigth);
    cudaDeviceSynchronize();
    int iteration = 0;
    do
    {
    	cout << iteration++ << endl;
    	hostHasBeenChanged = false;
    	HANDLE_ERROR(cudaMemcpy(hasAnyPixelBeenChanged, &hostHasBeenChanged, sizeof(bool), cudaMemcpyHostToDevice));
    	cout << "Enter function on device" << endl;
    	makeWatershade<< < grid, block >> >(cudaDifferenceImage, isFlooded, inputImage.cols, inputImage.rows, hasAnyPixelBeenChanged, treshold);
    	cudaDeviceSynchronize();
    	cout << "Leave function on device" << endl;
    	HANDLE_ERROR(cudaMemcpy(&hostHasBeenChanged, hasAnyPixelBeenChanged, sizeof(bool), cudaMemcpyDeviceToHost));
    	cout << "Changed: " << hostHasBeenChanged << endl;
    }while(hostHasBeenChanged);

    cout << "After flooding" << endl;
	HANDLE_ERROR(cudaFree(cudaDifferenceImage));

	markFloodedPixels<< < grid, block >> >(cudaInputImage, isFlooded, inputImage.cols, inputImage.rows);
    cudaDeviceSynchronize();
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&runningTime, startTime, stopTime);


    HANDLE_ERROR(cudaMemcpy(inputImage.data, cudaInputImage, cudaInputImageSize, cudaMemcpyDeviceToHost));

    cudaFreeHost(hasAnyPixelBeenChanged);
    cudaFree(cudaInputImage);
    cudaFree(isFlooded);
}

void ImageManager::saveImage(string outputPath)
{
    try
    {
	    imwrite(outputPath, inputImage);
    }
    catch (Exception &e)
    {
	    cout << "[ERROR] Could not save image" << endl << e.msg;
	    exit(-1);
    } 
}

void ImageManager::setBlockSize(int blockSize)
{
    this->blockSize = blockSize;
}

int ImageManager::getBlockSize()
{
    return this->blockSize;
}

void ImageManager::setGridSize(int gridSize)
{
    this->gridSize = gridSize;
}

int ImageManager::getGridSize()
{
    return this->gridSize;
}

void ImageManager::setStartPixel(Coords startCoords)
{
    startPixel = startCoords;
}

void ImageManager::setTreshold(int treshold)
{
    this->treshold = treshold;
}

void ImageManager::prepareOutputImage(Mat& inputImage)
{
    outputImage = Mat(inputImage.rows, inputImage.cols, CV_8UC3);
}

void ImageManager::clearRunningTime()
{
    runningTime = 0.0;
}
double ImageManager::getRunningTime()
{
    return runningTime;
}

void validateArguments(int argc, char** argv, string& inputPath, string& outputPath)
{
	if (argc < 3)
	{
		cout << "[ERR] Wrong count of parameters!" << endl;
		exit(-1);
	}
	inputPath = argv[1];
	outputPath = argv[2];
}

int main(int argc, char** argv) {
	Mat inputImage;
	string inputPath, outputPath;
	validateArguments(argc, argv, inputPath, outputPath);

    unique_ptr<ImageManager> imageManager;
    imageManager = make_unique<ImageManager>();
    

    Coords imgSize = imageManager->readImage(inputPath);

 /*
    Coords startCoords{0, 0};
    cout << "Rozmiar obrazka (szer, wys): " << imgSize.width << ", " << imgSize.heigth << ". Podaj piksel startowy (x, y): ";
    cin >> startCoords.width;
    cin >> startCoords.heigth;
    imageManager->setStartPixel(startCoords);

    int treshold;
    cout << "Podaj treshold zalania (0-255): ";
    cin >> treshold;
    imageManager->setTreshold(treshold);

	int size;
    cout << "Podaj rozmiar bloku (dlugosc jednego boku, rzeczywisty rozmiar bedzie wynosil kwadrat z podanej liczby): ";
    cin >> size;
    imageManager->setBlockSize(size);
    
    cout << "Podaj rozmiar gridu (dlugosc jednego boku, rzeczywisty rozmiar bedzie wynosil kwadrat z podanej liczby): ";
    cin >> size;
    imageManager->setGridSize(size);
    
 */
    Coords startCoords{100, 100};
    imageManager->setStartPixel(startCoords);
    imageManager->setTreshold(255);
    imageManager->setBlockSize(10);
    imageManager->setGridSize(10);

    cout << "grid size, block size, time" << endl;


    imageManager->clearRunningTime();
    imageManager->performWatershade();
    double time = imageManager->getRunningTime();
    cout << imageManager->getGridSize() * imageManager->getGridSize() << ", " << imageManager->getBlockSize() * imageManager->getBlockSize() << ", " << imageManager->getRunningTime();
    imageManager->saveImage(outputPath);


	return 0;
}
