#include "SVM_HOG_Training.h"

int main()
{
	char* PosSampleListDir = "./Resource/Pos/Ori_Pos.txt";
	char* NegSampleListDir = "./Resource/Neg/Ori_Neg_INRIA.txt";
	string PosSampleDir = "./Resource/Pos/";
	string NegSampleDir = "./Resource/Neg/";
	SVM_HOG_Training SVM(2416, 1218, PosSampleListDir, NegSampleListDir, PosSampleDir, NegSampleDir);

	char* SaveDir = "./NegWindow/";
	if (!SVM.dirExists(SaveDir))
	{
		cout << "We'll create a folder, " << SaveDir << endl;
		CreateDirectory(SaveDir, NULL);
	}

	/*It will generate 10 windows from Neg. samples by using feature matching to 
	distinguish the similarity among 10 windows. The time it take would be relative
	to the size of Neg. sample and how many windows you retrival.*/
	int result_NegWin = SVM.NegSampleWindowRetrieval_SIFT_SURF_SSIM(10, SaveDir);

	return result_NegWin;
}