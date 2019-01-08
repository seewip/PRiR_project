#pragma once
#include <string>
#include <opencv2/imgproc/imgproc.hpp>

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

    Coords readImage(std::string imagePath);  
    void saveImage(std::string outputPath);
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
    void prepareOutputImage(cv::Mat& inputImage);

    cv::Mat inputImage;
    cv::Mat outputImage;
    int blockSize;
    int gridSize;
    Coords startPixel;
    int treshold;
    float runningTime;
};