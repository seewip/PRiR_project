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
	outputImage = Mat(inputImage.rows, inputImage.cols, CV_8UC3);
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
    


    //Make grayscale from image
    unsigned char* cudaGrayscaleImage;
    HANDLE_ERROR(cudaMalloc((void**)& cudaGrayscaleImage, cudaGrayscaleImageSize));
    makeGrayscaleImage << < grid, block >> > (cudaInputImage, cudaGrayscaleImage, inputImage.cols, inputImage.rows, 1);

	cudaDeviceSynchronize();

	//Perform Gaussian blur
	unsigned char* cudaBlurredImage;
	HANDLE_ERROR(cudaMalloc((void**)& cudaBlurredImage, cudaGrayscaleImageSize));
    makeGaussianBlur << < grid, block >> > (cudaGrayscaleImage, cudaBlurredImage, inputImage.cols, inputImage.rows);
    cudaDeviceSynchronize();
    
    HANDLE_ERROR(cudaFree(cudaGrayscaleImage));
    
    //Make difference image (dilatated image - eroded image)
    unsigned char* cudaDifferenceImage;
    HANDLE_ERROR(cudaMalloc((void**)& cudaDifferenceImage, cudaGrayscaleImageSize));
    makeDifferenceImage<< < grid, block >> >(cudaBlurredImage, cudaDifferenceImage, inputImage.cols, inputImage.rows);
    
    cudaDeviceSynchronize();
    HANDLE_ERROR(cudaFree(cudaBlurredImage));



    //TU DZIALA

    brightenImage<< < grid, block >> >(cudaDifferenceImage, inputImage.cols, inputImage.rows, 3);
   /* saveGrayscaleImage<< < grid, block >> >(cudaDifferenceImage, cudaInputImage, inputImage.cols, inputImage.rows);
    HANDLE_ERROR(cudaMemcpy(outputImage.data, cudaInputImage, cudaInputImageSize, cudaMemcpyDeviceToHost));
    */

    bool* isFlooded;
    long cudaBooleanImageSize = sizeof(bool) * inputImage.rows * inputImage.cols;
    HANDLE_ERROR(cudaMalloc((void**)& isFlooded, cudaBooleanImageSize));

    bool* hasAnyPixelBeenChanged;
    HANDLE_ERROR(cudaMalloc((void**)& hasAnyPixelBeenChanged, sizeof(bool)));

    bool hostHasBeenChanged;
    HANDLE_ERROR(cudaMemset(isFlooded, 0, cudaBooleanImageSize));

    markStartPixel<<<1,1>>>(isFlooded, inputImage.cols, startPixel.width, startPixel.heigth);
    cudaDeviceSynchronize();

    do
    {
    	hostHasBeenChanged = false;
    	HANDLE_ERROR(cudaMemcpy(hasAnyPixelBeenChanged, &hostHasBeenChanged, sizeof(bool), cudaMemcpyHostToDevice));
    	makeWatershade<< < grid, block >> >(cudaDifferenceImage, isFlooded, inputImage.cols, inputImage.rows, hasAnyPixelBeenChanged, treshold);
    	cudaDeviceSynchronize();
    	HANDLE_ERROR(cudaMemcpy(&hostHasBeenChanged, hasAnyPixelBeenChanged, sizeof(bool), cudaMemcpyDeviceToHost));
    }while(hostHasBeenChanged);

	HANDLE_ERROR(cudaFree(cudaDifferenceImage));

	markFloodedPixels<< < grid, block >> >(cudaInputImage, isFlooded, inputImage.cols, inputImage.rows);
    cudaDeviceSynchronize();
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&runningTime, startTime, stopTime);

    HANDLE_ERROR(cudaMemcpy(outputImage.data, cudaInputImage, cudaInputImageSize, cudaMemcpyDeviceToHost));

    cudaFreeHost(hasAnyPixelBeenChanged);
    cudaFree(cudaInputImage);
    cudaFree(isFlooded);

}

void ImageManager::saveImage(string outputPath)
{
    try
    {
	    imwrite(outputPath, outputImage);
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

    cout << "grid size, block size, time" << endl;


    imageManager->clearRunningTime();
    imageManager->performWatershade();
    double time = imageManager->getRunningTime();
    cout << imageManager->getGridSize() * imageManager->getGridSize() << ", " << imageManager->getBlockSize() * imageManager->getBlockSize() << ", " << imageManager->getRunningTime();
    imageManager->saveImage(outputPath);


	return 0;
}
