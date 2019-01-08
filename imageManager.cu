#include "imageManager.hpp"
#include "cudaFunctions.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <string> 
#include <iostream>

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

    //Make image brighter alpha times
    double alpha = 3.0;
    brightenImage<< < grid, block >> >(cudaDifferenceImage, inputImage.cols, inputImage.rows, alpha);

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