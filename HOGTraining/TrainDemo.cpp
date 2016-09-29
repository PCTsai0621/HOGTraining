#include "SVM_HOG_Training.h"

int main()
{
	char* PosSampleListDir = "./Resource/Pos/Ori_Pos.txt";
	char* NegSampleListDir = "./Resource/Neg/Ori_Neg_INRIA.txt";
	string PosSampleDir = "./Resource/Pos/";
	string NegSampleDir = "./Resource/Neg/";
	char* NegWindowDir = "./Resource/NegWindow/";
	char* NegWindowListDir = "./Resource/NegWindow/NegWindow_List_INRIA.txt";
	SVM_HOG_Training SVM(2416, 1218, PosSampleListDir, NegSampleListDir, PosSampleDir, NegSampleDir);

	char* SaveDir = "./Train/";
	if (!SVM.dirExists(SaveDir))
	{
		cout << "We'll create a folder, " << SaveDir << endl;
		CreateDirectory(SaveDir, NULL);
	}

	/*It will create a model from Pos/Neg samples and then use the model to do hard example detection*/
  	int res_train = SVM.HOG_SVM_Train_iteration(false, 0, SaveDir, NegWindowListDir, NegWindowDir, 12180);

	/*It is allow you to do training with hardexample iteratively, which means that it will use trained
	model to do hard example detection and train again including these new hard exmaples*/
	//int res_train = SVM.HOG_SVM_Train_iteration(true, 5, SaveDir, NegWindowListDir, NegWindowDir, 12180);

	return res_train;
}