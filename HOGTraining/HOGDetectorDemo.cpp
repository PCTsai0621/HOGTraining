#include "SVM_HOG_Training.h"

int main()
{
	char* PosSampleListDir = "./Resource/Pos/Ori_Pos.txt";
	char* NegSampleListDir = "./Resource/Neg/Ori_Neg_INRIA.txt";
	string PosSampleDir = "./Resource/Pos/";
	string NegSampleDir = "./Resource/Neg/";
	SVM_HOG_Training SVM(2416, 1218, PosSampleListDir, NegSampleListDir, PosSampleDir, NegSampleDir);

	ifstream fin("./Detector_2416Pos_12180Neg_3940HardEx_0Iteration.txt");
	vector<float> detector;
	vector<int> miss;
	float val;

	/*load detector*/
	while (!fin.eof())
	{
		fin >> val;
		detector.push_back(val);
	}
	fin.close();

	/*You can load the existing trained model to detect video. If you want to get the 
	detector after training, use "getDetector()". */
	int result_detector = SVM.HOGDetector(detector, "./", miss);

	return result_detector;
}