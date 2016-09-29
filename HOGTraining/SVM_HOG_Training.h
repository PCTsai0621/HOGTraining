#pragma once

#include "Common.h"

#define STATUS_OK 1
#define STATUS_ERROR 0

#define XML_DEFAULT_DIR "SVM_HOG_2400PosINRIA_12000Neg_HardExample.xml"
#define DEBUG_MODE true
#define PRINT_DETAILLY true
#define IGNORE false

using namespace std;
using namespace cv;
using namespace cv::ml;

/*
This class includes the folowing functions:
**Crop the negative sample windows from negative sample
**Set HOG detector and detecting the video
**Train the Detector from Pos/Neg sample and hardexample 
  and detect the hard example from negative sample.
*/

class SVM_HOG_Training
{

private:
	int result_train,			//result of HOG+SVM training
		result_HEDetect,		//result of hard example detection
		result_HOGDetector,		//result of setting HOG detector
		result_NSWRetrieval,	//result of negative sample windows retrieval
		PosNumber,				//number of positive sample
		NegNumber,				//number of negative sample
		CentralCrop,			//Whether to crop positive sample
		IterationTimes,			//Times of Traing Iteration
		IterationCount,			//Traing Iteration Counter
		_HardExNumber_,			//Hard example number for iteration
		_MissDetectCount_;		//number of miss detect count

	string TrainXmlDir			//directory of xml file used to load the information of detector
		, PosSampleDir			//directory of positive sample
		, NegSampleDir			//directory of negative sample
		, HardExDir;			//directory of hard example 

	char* PosListDir;			//directory of positive sample list
	char* NegListDir;			//directory of negative sample list
	char* HardExListDir;		//directory of hard example list

	bool IterationOrNot;		//Whether to train iteratively 
	vector<float> DetectorTemp;	//Temp of detector for training iteratively

	/*Variable for HOGDetector inline function*/
	int64 hog_work_begin;		//time when hog work in HOGDetector begin
	double hog_work_fps;		//FPS of hog work in HOGDetector
	int64 work_begin;			//time when HOGDetector work begin
	double work_fps;			//FPS of HOGDetector work 


public:
	/*
	This class includes the functions of retrievaling the negative sample windows from negative sample,
	setting HOG detector and detecting the video, training the Detector from Pos/Neg sample and hard
	example and detecting the hard example from negative sample.
	@_PosNumber: Number of positive sample
	@_NegNumber: Number of negative sample
	@_PosSampleListDir: Directory of positive sample list
	@_NegSampleListDir: Directory of negative sample list
	@_PosSampleDir: Directory of positive sample
	@_NegSampleDir: Directory of negative sample
	*/
	SVM_HOG_Training(int _PosNumber,
		int _NegNumber,
		char* _PosListDir,
		char* _NegListDir,
		string _PosSampleDir,
		string _NegSampleDir);

	SVM_HOG_Training();
	~SVM_HOG_Training();

	/*
	Retrievaling the negative sample windows from negative sample.
	@_WindowsNumber: Number of negative sample windows(64x128) which is cut from negative sample
	@_SaveDir: Directory to save negative sample windows
	*/
	int NegSampleWindowRetrieval(int _WindowsNumber,
		char* _SaveDir);

	/*
	Training the Detector from Pos/Neg sample and hard example.
	If you want to use your own trained xml file for detector, use "setTrainXmlDir" function to load.
	@_TrainOrNot: Whether to train detector or not. If not, it will use default detector.
	@_HardExNumber: Number of hard example
	@_HardExListDir: Directory of hard example list
	@_HardExDir: Directory of hard example
	@_UseHardExOrNot: Whether to use hard example for training
	@_SaveDir: Directory to save svm xml file and detector
	@_NegWindowsListDir: Directory of negative sample windows list
	@_NegWindowsDir: Directory of negative sample windows
	@_NegWindowsNumber: Number of negative sample windows
	**Default HOG size is 64*128
	*/
	int HOG_SVM_Train(bool _TrainOrNot,
		int _HardExNumber,
		string _HardExListDir,
		string _HardExDir,
		bool _UseHardExOrNot,
		char* _SaveDir,
		char* _NegWindowsListDir,
		char* _NegWindowsDir,
		int _NegWindowsNumber);

	/*
	Detecting the hard example from negative sample and save.
	@_SaveDir: Directory to save hard examples
	@_DetectorDir: Directory of trained detector in txt file
	*/
	int HardExampleDetect(char* _SaveDir,
		char* _DetectorDir);

	/*
	Detecting the hard example from negative sample and save.
	@_SaveDir: Directory to save hard examples
	@_Detector: Vector of detector
	*/
	int HardExampleDetect(char* _SaveDir,
		vector<float> _Detector);



	/*Setting HOG detector and detecting the video then saving the video*/
	int HOGDetector(vector<float> _Detector,
		char* _SaveDir, vector<int> &_MissDetectorList);

	/*Inline function for HOGDetector*/
	void hogWorkBegin();
	void hogWorkEnd();
	string hogWorkFps() const;
	void workBegin();
	void workEnd();
	string workFps() const;

	/*
	Training Iteratively with the iteration of hard example.
	If _IterationTimes is 0, it means that it will train with negative window, positive sample and
	hard example(first times).
	@_IterationOrNot: Whether to train iteratively
	@_IterationTimes: Times of iteration
	@_SaveDir: Directory to save svm xml file, detector txt file and hard example
	@_NegWindowsListDir: Directory of negative sample windows list
	@_NegWindowsDir: Directory of negative sample windows
	@_NegWindowsNumber: Number of negative sample windows
	*/
	int HOG_SVM_Train_iteration(bool _IterationOrNot,
		int _IterationTimes,
		char* _SaveDir,
		char* _NegWindowsListDir,
		char* _NegWindowsDir,
		int _NegWindowsNumber);

	/*Getting detector vector*/
	vector<float> getDetector();

	/*
	Checking Directory isExists.
	If return 0, the path must be wrong or not exist.
	@dirName_in: The Directory you want to check
	*/
	bool dirExists(const char* dirName_in);

	/*Creating hard example list for iteration*/
	int HardExampleIterationList(char* _SaveDir,
		int _FolderCount,
		int _HardExNumberTemp);

	/*
	Calculating the SSIM of two input image
	@_MatrixReference: Input image you want to compare with
	@_MatrixUnderTest: Input image you want to compare
	*/
	cv::Scalar calMSSIM(const Mat& _MatrixReference,
		const Mat& _MatrixUnderTest);

	/*
	Calculating the PSNR of two input image
	@_MatrixReference: Input image you want to compare with
	@_MatrixUnderTest: Input image you want to compare
	*/
	double calPSNR(const Mat& _MatrixReference,
		const Mat& _MatrixUnderTest);

	/*Test PSNR SSIM ability of distingushing negative sample*/
	int PSNR_SSIMTestDemo();

	/*Feature Matching with FLANN and SURF*/
	int FeatureMatchingFLANN(cv::Mat _InputImage1,
		cv::Mat _InputImage2);

	/*FLANNDemo*/
	int FLANNDemo();

	/*Use SURF descriptor and SSIM to find similar structure of two input image*/
	int FeatureHomography(cv::Mat _InputImage1,
		cv::Mat _InputImage2);

	/*Compute keypoints by SIFT detector*/
	std::vector<cv::KeyPoint> KeyPointTest_SIFT(cv::Mat _InputImage);

	/*Find the homography of the two input image*/
	int SVM_HOG_Training::HomographyTest_SIFT_DAISY(cv::Mat _InputImage1,
		cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2);

	/*Find the homography of the two input image*/
	int SVM_HOG_Training::HomographyTest_SURF_SURF(cv::Mat _InputImage1,
		cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2);

	/*Find the homography of the two input image*/
	int SVM_HOG_Training::HomographyTest_SURF_SIFT(cv::Mat _InputImage1,
		cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2);

	/*Find the homography of the two input image*/
	int SVM_HOG_Training::HomographyTest_SIFT_SURF(cv::Mat _InputImage1,
		cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2, std::vector<int> &_BoxSize);

	/*NegWinSimilarTest with SIFT+SURF+SSIM*/
	int NegSampleWindowRetrieval_SIFT_SURF_SSIM(int _WindowsNumber,
		char* _SaveDir);

	/*NegWinSimilarTest with SIFT+SURF+SSIM for multithread version*/
	int NegSampleWindowRetrieval_SIFT_SURF_SSIM_M(int _WindowsNumber,
		char* _SaveDir, vector<string> _ImgList);

	/*Homography Test for the result of NegWinSimilarTest with SIFT+SURF+SSIM*/
	int NegSampleWindowRetrievalHomoTest(char* _NegWindowsDir, char* _NegWindowsListDir);

	/*SVM Training Demo*/
	void SVM_Train_Demo();

	/*SIFT+DAISY Demo*/
	void SIFT_DAISY_Demo();

	/*Homo with SIFT+SURF Demo*/
	void HomographyTest_SIFT_SURF_Demo();
};

