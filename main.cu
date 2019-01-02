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
    void setTreshold(unsigned char treshold);
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
    unsigned char treshold;
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
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;  //TODO:sprawdz jak to obliczyc!
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;  //TODO:sprawdz jak to obliczyc!
    
    double redWeigth = 0.33;    
    double greenWeigth = 0.34;
    double blueWeigth = 0.33;

    double currentColor;
    int inputIndex;
    for(xCoord; xCoord < xSize; xCoord += blockDim.x)
    {
        for(yCoord; yCoord < ySize; yCoord += blockDim.y)
        {
            currentColor = 0.0;
			inputIndex = xSize * yCoord * 3 + xCoord * 3;

			currentColor += imageIn[inputIndex + 2] * redWeigth;
			currentColor += imageIn[inputIndex + 1] * greenWeigth;
			currentColor += imageIn[inputIndex] * blueWeigth;
            imageOut[xSize * yCoord + xCoord] = (unsigned char)currentColor;
	    }
    }
}

__global__ void makeDifferenceImage(unsigned char* grayscaleImage, unsigned char* differenceImage, long xSize, long ySize)
{
    long xCoord = blockIdx.x * blockDim.x + threadIdx.x;  //TODO:sprawdz jak to obliczyc!
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y;  //TODO:sprawdz jak to obliczyc!
    
    if(xCoord == 0 or xCoord == xSize-1)
    {
        differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
        xCoord += blockDim.x;
    }
    if(yCoord == 0 or xCoord == xSize-1)
    {
        differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
        yCoord += blockDim.y;
    }

    unsigned char brightestPoint;
    unsigned char darkestPoint;
    for(xCoord; xCoord < xSize; xCoord+= blockDim.x)
    {   
        if(xCoord == xSize-1)
        {
            differenceImage[xSize * yCoord + xCoord] = grayscaleImage[xSize * yCoord + xCoord];
            break;
        }
        for(yCoord; yCoord < ySize; yCoord+= blockDim.y)
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

    long xCoord = blockIdx.x * blockDim.x + threadIdx.x; //TODO:sprawdz jak to obliczyc!
    long yCoord = blockIdx.y * blockDim.y + threadIdx.y; //TODO:sprawdz jak to obliczyc!

    while(xCoord <= 1)
    {
        xCoord += blockDim.x;
    }
    while(yCoord <= 1)
    {
        yCoord += blockDim.y;
    }
    
    long inputIndex, outputIndex;
    unsigned char currentColor;
    
    //blur grayscale image
    for(xCoord; xCoord < xSize-2; xCoord+= blockDim.x)
    {
        for(yCoord; yCoord < ySize-2; yCoord+= blockDim.y)
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
        }
    }
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
    cudaMalloc((void**)& cudaInputImage, cudaInputImageSize); //TODO error check
    
    if (cudaMemcpy(cudaInputImage, inputImage.data, cudaInputImageSize, cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout << "[ERROR] Could not copy data from CPU to GPU" << endl;
	    cudaFree(cudaInputImage);
	    exit(-1);
    }
    
    unsigned char* cudaGrayscaleImage;
    cudaMalloc((void**)& cudaGrayscaleImage, cudaGrayscaleImageSize); //TODO error check
    
    

    


    makeGrayscaleImage << < grid, block >> > (cudaInputImage, cudaGrayscaleImage, inputImage.cols, inputImage.rows);
    
    //TODO synchronizacja
    
    unsigned char* cudaBlurredImage;
    cudaMalloc((void**)& cudaBlurredImage, cudaGrayscaleImageSize); //TODO error check
    
    makeGaussianBlur << < grid, block >> > (cudaGrayscaleImage, cudaBlurredImage, inputImage.cols, inputImage.rows);
    
    //TODO synchronizacja

    cudaFree(cudaGrayscaleImage);
    unsigned char* cudaDifferenceImage;
    cudaMalloc((void**)& cudaDifferenceImage, cudaGrayscaleImageSize); //TODO error check
    
    makeDifferenceImage<< < grid, block >> >(cudaBlurredImage, cudaDifferenceImage, inputImage.cols, inputImage.rows);
    
    //TODO synchronizacja
    cudaFree(cudaBlurredImage);

    bool* cudaIsPixelFlooded;
    long cudaBooleanImageSize = sizeof(bool) * inputImage.rows * inputImage.cols;
    cudaMalloc((void**)& cudaIsPixelFlooded, cudaBooleanImageSize); //TODO error check

    //TODO makeWatershade
    //TODO synchronizacja
    //TODO markFloodedPixel
    //TODO synchronizacja


    cudaDeviceSynchronize();
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&runningTime, startTime, stopTime);


    bool errorWhileCopying = false;
    if (cudaMemcpy(cudaInputImage, inputImage.data, cudaInputImageSize, cudaMemcpyHostToDevice) != cudaSuccess)
    {
	    cout << "[ERROR] Could not copy data from GPU to CPU" << endl;
	    errorWhileCopying = true;
    }
    cudaFree(cudaInputImage);
    cudaFree(cudaDifferenceImage);
    cudaFree(cudaIsPixelFlooded);

    if(errorWhileCopying)
    {
        exit(-1);
    }
}

void ImageManager::saveImage(string outputPath)
{
    try
    {
	    imwrite(outputPath, outputImage);
    }
    catch (Exception &e)
    {
	    cout << "[ERROR] Could not save blured image" << endl << e.msg;
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

void ImageManager::setTreshold(unsigned char treshold)
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
    int size;
	validateArguments(argc, argv, inputPath, outputPath);

    unique_ptr<ImageManager> imageManager;
    imageManager = make_unique<ImageManager>();
    

    Coords startCoords{0, 0};
    Coords imgSize = imageManager->readImage(inputPath);
    cout << "Rozmiar obrazka (szer, wys): " << imgSize.width << ", " << imgSize.heigth << ". Podaj piksel startowy (x, y): ";
    cin >> startCoords.width;
    cin >> startCoords.heigth;
    imageManager->setStartPixel(startCoords);

    unsigned char treshold;
    cout << "Podaj treshold zalania (0-255): ";
    cin >> treshold;
    imageManager->setTreshold(treshold);

    cout << "Podaj rozmiar bloku (dlugosc jednego boku, rzeczywisty rozmiar bedzie wynosil kwadrat z podanej liczby): ";
    cin >> size;
    imageManager->setBlockSize(size);
    
    cout << "Podaj rozmiar gridu (dlugosc jednego boku, rzeczywisty rozmiar bedzie wynosil kwadrat z podanej liczby): ";
    cin >> size;
    imageManager->setGridSize(size);
    

    cout << "grid size, block size, time" << endl;


    imageManager->clearRunningTime();
    imageManager->performWatershade();
    double time = imageManager->getRunningTime();
    cout << imageManager->getGridSize() * imageManager->getGridSize() << ", " << imageManager->getBlockSize() * imageManager->getBlockSize() << ", " << imageManager->getRunningTime();
    imageManager->saveImage(outputPath);


	return 0;
}
