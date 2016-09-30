# HOGTraining
This project implements the following paper:
```
Navneet Dalal, Bill Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005
``` 
It provides a training framework by OpenCV API.

## Platform
This project is developed on Windows. If you want use it on other platform, you need to modify the filesystem corresponding to your platform.

## Dependencies
- OpenCV310 (xfeature module)

## Dataset
The training dataset can download from [INRIA Person Dataset](http://pascal.inrialpes.fr/data/human/).
 - Positive training dataset: 2416 photos of 96x160 pixels
 - Negative training dataset: 1218 photos with no fixed size

## Usage
### Data Process
- The positive training set can be found in folder `train_64x128_H96\pos\`. The algorithm have implemented the normalization for positive training set, that is, crop the image of 64x128 pixels centered on the person.
- For normalization, use `NegSampleWindowRetrieval_SIFT_DAISY_SSIM()` to sample 10 samples (also called negative windows) of 64x128 pixels from each negative training data. The details is shown in `NegWindowRetrivalDemo.cpp`.

### Training
After sampling the negative windows, you can see how to use positive training set and negative windows to train a model. The details is shown in `TrainDemo.cpp`. It is also allowed to do training with hard examples iteratively, which means that it uses the trained model to do hard example detection and do training again including these new hard exmaples. Finally, it will generate trained model `Model.txt`. 

### Detection
In `HOGDetectorDemo.cpp`, it shows how to load trained model and detect on video.


