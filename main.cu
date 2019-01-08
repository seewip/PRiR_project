#include <iostream>
#include <memory>
#include <string> 
#include "imageManager.hpp"

#define tests

using namespace std;

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

#ifdef presentation
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
    cout << imageManager->getGridSize() * imageManager->getGridSize() << ", "
    	 << imageManager->getBlockSize() * imageManager->getBlockSize() << ", "
    	 << imageManager->getRunningTime() << endl;
#endif

#ifdef tests
    cout << "grid size, block size, time" << endl;
    for(int i=1; i<33; i++)
    {
    	for(int j=1; j<33; j++)
    	{
    		imageManager->setBlockSize(i);
    		imageManager->setGridSize(j);
    		imageManager->clearRunningTime();
			imageManager->performWatershade();
			double time = imageManager->getRunningTime();
			cout << imageManager->getGridSize() * imageManager->getGridSize() << ", "
				 << imageManager->getBlockSize() * imageManager->getBlockSize() << ", "
				 << imageManager->getRunningTime() << endl;
    	}
    }
#endif

    imageManager->saveImage(outputPath);

	return 0;
}
