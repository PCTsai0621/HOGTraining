#include "SVM_HOG_Training.h"

SVM_HOG_Training::SVM_HOG_Training(int _PosNumber,
	int _NegNumber,
	char* _PosListDir,
	char* _NegListDir,
	string _PosSampleDir,
	string _NegSampleDir)
{
	PosNumber = _PosNumber;
	NegNumber = _NegNumber;
	PosListDir = _PosListDir;
	NegListDir = _NegListDir;
	PosSampleDir = _PosSampleDir;
	NegSampleDir = _NegSampleDir;

	CentralCrop = 1;
	IterationOrNot = false;				//default training once
	IterationTimes = 0;					//default not training iteratively
	IterationCount = 0;
}

SVM_HOG_Training::SVM_HOG_Training()
{
}


SVM_HOG_Training::~SVM_HOG_Training()
{
}

int SVM_HOG_Training::HOG_SVM_Train(bool _TrainOrNot,
	int _HardExNumber,
	string _HardExListDir,
	string _HardExDir,
	bool _UseHardExOrNot,
	char* _SaveDir,
	char* _NegWindowsListDir,
	char* _NegWindowsDir,
	int _NegWindowsNumber)
{

	Ptr<SVM> mySVM = cv::ml::SVM::create();

	/*
	Use to compute HOG descriptor. Specification as bel ow:
	detector window: (64 ,128)
	block size: (16, 16)
	block stride: (8, 8)
	cell size: (8, 8)
	number of histogram bin: 9
	*/
	HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	int DescriptorDim;			//the dimension of HOGDescriptor

	if (_TrainOrNot)
	{
		string ImgNameTemp;		//load directory of Pos/Neg/HardEx image

		ifstream finPos(PosListDir);	//positive sample directionary list
		ifstream finNegWindow(_NegWindowsListDir);	//negative sample directionary list

		/*
		Matrix composed of all feature vector of samples.
		column = number of all samples
		row = dimension of HOGDescriptor
		*/
		Mat sampleFeatureMat;

		/*
		Vector composed of the class of the samples.
		1 for poeple, -1 for scene(no peoeple)
		*/
		Mat sampleLabelMat;

		/*Read positive samples and compute hog feature*/
		for (int num = 0; num<PosNumber && getline(finPos, ImgNameTemp); num++)
		{

			if (PRINT_DETAILLY) std::cout << "Compute HOG, PosImg " << num + 1 << "：" << ImgNameTemp << endl;
			ImgNameTemp = PosSampleDir + ImgNameTemp;

			Mat src = imread(ImgNameTemp);		//load image  
			if (!src.data)                      // Check for invalid input
			{
				cout << "Could not open or find the image in PosImg " << num + 1 << endl;
				cout << "Please check the diectory is incorrect." << endl;
				cout << ImgNameTemp << endl;
				return STATUS_ERROR;
			}

			if (CentralCrop)					//crop the pos image to 64x128 with margin 16 pixels
				src = src(Rect(16, 16, 64, 128));

			vector<float> descriptors;			//HOG descriptor vector  

 			hog.compute(src, descriptors, Size(8, 8));	//Compute HOG descriptor with detect window stride (8,8)
			
			//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵  
			if (0 == num)
			{
				DescriptorDim = descriptors.size();
				//Initialize matrix composed of all feature vector of samples
				sampleFeatureMat = Mat::zeros(PosNumber + _NegWindowsNumber + _HardExNumber, DescriptorDim, CV_32FC1);
				//Initialize vector composed of the class of the samples  
				sampleLabelMat = Mat::zeros(PosNumber + _NegWindowsNumber + _HardExNumber, 1, CV_32SC1);
			}

			//Save HOG descriptor in sampleFeatureMat  
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num, i) = descriptors[i];//第num个样本的特征向量中的第i个元素  
			sampleLabelMat.at<float>(num, 0) = 1;//正样本类别为1，有人  

		}

		cout << "Loading negative samples..." << endl;
		if (DEBUG_MODE) system("pause");

		/*Read negative samples and compute hog feature*/
		for (int num = 0; num < _NegWindowsNumber && getline(finNegWindow, ImgNameTemp); num++)
		{
			if (PRINT_DETAILLY) std::cout << "Compute HOG, NegWin " << num + 1 << "：" << ImgNameTemp << endl;
			ImgNameTemp = _NegWindowsDir + ImgNameTemp;
			Mat src = imread(ImgNameTemp);
			if (!src.data)                      // Check for invalid input
			{
				cout << "Could not open or find the image in NegImg " << num + 1 << endl;
				cout << "Please check the diectory is incorrect." << endl;
				cout << ImgNameTemp << endl;
				return STATUS_ERROR;
			}

			vector<float> descriptors;
			hog.compute(src, descriptors, Size(8, 8));

			//Save HOG descriptor in sampleFeatureMat  
			for (int i = 0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num + PosNumber, i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素  
			sampleLabelMat.at<float>(num + PosNumber, 0) = -1;//负样本类别为-1，无人  

		}

		/*Read hard examples and compute hog feature*/
		if (_UseHardExOrNot)
		{
			cout << "Loading hard examples..." << endl;
			if (DEBUG_MODE) system("pause");

			ifstream finHardExample(_HardExListDir);	//HardExample负样本的文件名列表
			for (int num = 0; num<_HardExNumber && getline(finHardExample, ImgNameTemp); num++)
			{
				if (PRINT_DETAILLY) std::cout << "Compute HOG, HardEx " << num + 1 << "：" << ImgNameTemp << endl;
				ImgNameTemp = _HardExDir + ImgNameTemp;
				Mat src = imread(ImgNameTemp);
				if (!src.data)                      // Check for invalid input
				{
					cout << "Could not open or find the image in HardEx " << num + 1 << endl;
					cout << "Please check the diectory is incorrect." << endl;
					cout << ImgNameTemp << endl;
					return STATUS_ERROR;
				}

				vector<float> descriptors;
				hog.compute(src, descriptors, Size(8, 8));

				//Save HOG descriptor in sampleFeatureMat
				for (int i = 0; i<DescriptorDim; i++)				//第PosSamNO+num个样本的特征向量中的第i个元素
					sampleFeatureMat.at<float>(num + PosNumber + _NegWindowsNumber, i) = descriptors[i];
				sampleLabelMat.at<float>(num + PosNumber + _NegWindowsNumber, 0) = -1;//负样本类别为-1，无人
			}
		}

		cout << "Ready for training..." << endl;
		if (DEBUG_MODE) system("pause");

		/*Train SVM classifier*/
		char XmlSaveDirTemp[256];
		
		mySVM->setType(SVM::C_SVC);
		mySVM->setKernel(SVM::LINEAR);
		mySVM->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1.1920928955078125e-7F));
		mySVM->setC(1.0000000000000000e-2F);
		
		Ptr<cv::ml::TrainData> TrainData = cv::ml::TrainData::create(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLabelMat);

		std::cout << "Start to Train classifier..." << endl;
		mySVM->train(sampleFeatureMat, cv::ml::ROW_SAMPLE, sampleLabelMat);//训练分类器
		std::cout << "Train successfully" << endl;

		/*save svm*/

		//When you choose to iteration, it will print out every iteration times
		if (IterationOrNot)
		{
			if ((_UseHardExOrNot) && (IterationTimes >= 0))
			{
				sprintf_s(XmlSaveDirTemp, sizeof(XmlSaveDirTemp), "%s%dPos_%dNeg_%dHardEx_%dIteration.xml", _SaveDir,
					PosNumber, _NegWindowsNumber, _HardExNumber, IterationCount);
				mySVM->save(XmlSaveDirTemp);//将训练好的SVM模型保存为xml文件
				cout << "save classifier in: " << XmlSaveDirTemp << endl;
			}
		}
		else
		{
			sprintf_s(XmlSaveDirTemp, sizeof(XmlSaveDirTemp), "%s%dPos_%dNeg_%dHardEx_0Iteration.xml", _SaveDir,
				PosNumber, _NegWindowsNumber, _HardExNumber);
			mySVM->save(XmlSaveDirTemp);//将训练好的SVM模型保存为xml文件
			cout << "save classifier in: " << XmlSaveDirTemp << endl;
		}
	}
	else
	{
		/*load svm*/
		 
		mySVM->load(XML_DEFAULT_DIR);//从XML文件读取训练好的SVM模型  
	}

	cout << "Ready to load detector" << endl;
	cout << "Please press any key to continue" << endl;
	if (DEBUG_MODE) system("pause");

	/*
	SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个行向量，将该向量前面乘以-1。之后，再该行向量的最后添加一个元素rho。
	如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	*/
	//获取支持向量机：矩阵默认是CV_32F
	Mat SupportVector = mySVM->getSupportVectors();//

	//获取alpha和rho
	Mat alpha;//每个支持向量对应的参数α(拉格朗日乘子)，默认alpha是float64的
	Mat svIndex;//支持向量所在的索引
	float rho = mySVM->getDecisionFunction(0, alpha, svIndex);

	//转换类型:这里一定要注意，需要转换为32的
	Mat alpha2;
	alpha.convertTo(alpha2, CV_32FC1);

	//结果矩阵，两个矩阵相乘
	Mat result(1, 3780, CV_32FC1);
	result = alpha2*SupportVector;

	vector<float> myDetector;

	//乘以-1，这里为什么会乘以-1？
	//注意因为svm.predict使用的是alpha*sv*another-rho，如果为负的话则认为是正样本，在HOG的检测函数中，使用rho+alpha*sv*another(another为-1)
	for (int i = 0; i < 3780; ++i)
	{
		result.at<float>(0, i) *= -1;
		myDetector.push_back(result.at<float>(0, i));		//save as a vector
	}
	myDetector.push_back(rho);							//save as a vector, finally in size of 3781

	DetectorTemp.assign(myDetector.begin(), myDetector.end());


	std::cout << "The dimension of detector ：" << myDetector.size() << endl;
	if (DEBUG_MODE) std::cout << "---DEBUG LINE3---" << endl;

	/*Setting HOGDescriptor for HOG detector*/
	HOGDescriptor myHOG(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
	myHOG.setSVMDetector(myDetector);

	if (DEBUG_MODE) std::cout << "---DEBUG LINE4---" << endl;

	//Save parameter of detector in txt file  
	if (_TrainOrNot)
	{
		char detectorTXTTemp[1000];
		if (IterationOrNot)
		{
			if ((_UseHardExOrNot) && (IterationTimes >= 0))
			{
				sprintf_s(detectorTXTTemp, sizeof(detectorTXTTemp), "%sDetector_%dPos_%dNeg_%dHardEx_%dIteration.txt",
					_SaveDir, PosNumber, _NegWindowsNumber, _HardExNumber, IterationCount);
				//mySVM.save(detectorTXTTemp);//将训练好的SVM模型保存为xml文件
				ofstream fout(detectorTXTTemp, ofstream::trunc);
				for (int i = 0; i<myDetector.size(); i++)
				{
					fout << myDetector[i];
					if (i != myDetector.size() - 1)
					{
						fout << endl;
					}
				}
				cout << "Detector have saved in: " << detectorTXTTemp << endl;
			}
		}
		else
		{
			sprintf_s(detectorTXTTemp, sizeof(detectorTXTTemp), "%sDetector_%dPos_%dNeg_%dHardEx_0Iteration.txt",
				_SaveDir, PosNumber, _NegWindowsNumber, _HardExNumber);
			//mySVM.save(detectorTXTTemp);//将训练好的SVM模型保存为xml文件
			ofstream fout(detectorTXTTemp, ofstream::trunc);
			for (int i = 0; i<myDetector.size(); i++)
			{
				fout << myDetector[i];
				if (i != myDetector.size() - 1)
				{
					fout << endl;
				}
			}
			cout << "Detector have saved in: " << detectorTXTTemp << endl;
		}

		cout << "Please press any key to continue" << endl;
		if (DEBUG_MODE) system("pause");
	}

	//if not training iteratively, input a image to test detector
	if (!IterationOrNot)
	{
		Mat src = imread("ImageTest.jpg");
		imshow("test", src);
		vector<Rect> found, found_filtered;  
		if (DEBUG_MODE) std::cout << "---DEBUG LINE5---" << endl;
		std::cout << "HOG multiscale detection test" << endl;
		myHOG.detectMultiScale(src, found, 0, Size(8, 8), Size(32, 32), 1.05, 0);
		std::cout << "number of detect rect: " << found.size() << endl;
		if (DEBUG_MODE) std::cout << "---DEBUG LINE6---" << endl;

		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中  
		for (int i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			int j = 0;
			for (; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);
		}

		//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整  
		for (int i = 0; i<found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(src, r.tl(), r.br(), Scalar(0, 255, 0), 3);
		}

		imwrite("ImageTest_detected.jpg", src);
		namedWindow("src", 0);
		imshow("src", src);
		waitKey(0);						//注意：imshow之后必须加waitKey，否则无法显示图像
	}

	return STATUS_OK;
}

int SVM_HOG_Training::NegSampleWindowRetrieval(int _WindowsNumber,
	char* _SaveDir)
{
	Mat src;
	string ImgNameTemp;
	char NegSampleWindowSaveName[1000];				//Temp for name of negative sample window 
	char NegSampleWindowListSaveName[1000];
	int lineCount = 0;
	ifstream finNeg(NegListDir);					//Load negative sample list

	char NegSampleWindowList[1000];
	sprintf_s(NegSampleWindowList, sizeof(NegSampleWindowList), "%sINRIANegWindowsList.txt", _SaveDir);
	ofstream foutNegWinList(NegSampleWindowList);

	while (getline(finNeg, ImgNameTemp))
	{
		if (PRINT_DETAILLY) std::cout << "Cut NegWindow, NegImg " << lineCount + 1 << "：" << ImgNameTemp << endl;
		ImgNameTemp = NegSampleDir + ImgNameTemp;
		src = imread(ImgNameTemp);
		if (!src.data)                      // Check for invalid input
		{
			cout << "Could not open or find the image in NegImg " << lineCount + 1 << endl;
			cout << "Please check the diectory is incorrect." << endl;
			cout << ImgNameTemp << endl;
			return STATUS_ERROR;
		}

		//图片大小应该能能至少包含一个64*128的窗口  
		if (src.cols >= 64 && src.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子  

			//从每张图片中随机裁剪10个64*128大小的不包含人的负样本  
			for (int i = 0; i<_WindowsNumber; i++)
			{
				int x = (rand() % (src.cols - 64)); //左上角x坐标  
				int y = (rand() % (src.rows - 128)); //左上角y坐标  
				Mat imgROI = src(Rect(x, y, 64, 128));
				sprintf_s(NegSampleWindowSaveName, sizeof(NegSampleWindowSaveName), "%snoperson%06d.jpg",
					_SaveDir, ++lineCount);//生成裁剪出的负样本图片的文件名 
				sprintf_s(NegSampleWindowListSaveName, sizeof(NegSampleWindowListSaveName), "noperson%06d.jpg",
					lineCount);
				if (PRINT_DETAILLY) cout << NegSampleWindowSaveName << endl;
				imwrite(string(NegSampleWindowSaveName), imgROI);//保存文件  
				foutNegWinList << NegSampleWindowListSaveName << endl;
			}
		}

	}

	return STATUS_OK;
}

int SVM_HOG_Training::HardExampleDetect(char* _SaveDir,
	char* _DetectorDir)
{
	Mat src;
	char saveName[1000];//剪裁出来的hard example图片的文件名  
	string ImgNameTemp;
	ifstream finDetector(_DetectorDir);				//Load trained detector
	ifstream finNeg(NegListDir);					//Load negative sample list 
	int lineCount = 0;								//Line number for reading detector txt file
	int HardExDetectCount = 0;						//Number for counting total hard example

	/*Save hard example list*/
	char HardExListDir[1000];						//Directory of hard example list to save
	sprintf_s(HardExListDir, "%sHardExList.txt", _SaveDir);
	ofstream foutList(HardExListDir, ofstream::trunc);

	float tempSVMParam;								//Read in SVM parameter from detector file
	vector<float> myDetector;						//The dimension should be 3781  
	while (!finDetector.eof())
	{
		finDetector >> tempSVMParam;
		myDetector.push_back(tempSVMParam);			//Put SVM parameter in to detector  
	}

	if (myDetector.size() != 3781)
	{
		cout << "The size of the detector is incorrect: " << myDetector.size() << endl;
		cout << "Please check your detector." << endl;
		return STATUS_ERROR;
	}

	HOGDescriptor hog;
	hog.setSVMDetector(myDetector);					//Set HOG detector  

	while (getline(finNeg, ImgNameTemp))
	{
		if (PRINT_DETAILLY) std::cout << "Detect HardExample, NegImg " << lineCount + 1 << "：" << ImgNameTemp << endl;
		ImgNameTemp = NegSampleDir + ImgNameTemp;
		src = imread(ImgNameTemp);
		if (!src.data)								// Check for invalid input
		{
			cout << "Could not open or find the image in NegImg " << lineCount + 1 << endl;
			cout << "Please check the diectory is incorrect." << endl;
			cout << ImgNameTemp << endl;
			return STATUS_ERROR;
		}

		int HardExDetectPer = 0;					//Count number of hard example in every negative sample

		Mat HardExTemp = src.clone();				//Avoid to modify origin negative sample
		vector<Rect> foundRect;						//Save all hard example box
		hog.detectMultiScale(src, foundRect, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		HardExDetectPer = foundRect.size();

		for (int i = 0; i < foundRect.size(); i++)
		{
			if (HardExDetectCount != 0)				//foutList line jump after the first line is saved
			{
				foutList << endl;
			}

			/*Force box boundary not be out of the boundary of NegSample*/
			Rect foundRectTemp = foundRect[i];
			if (foundRectTemp.x < 0)
				foundRectTemp.x = 0;
			if (foundRectTemp.y < 0)
				foundRectTemp.y = 0;
			if (foundRectTemp.x + foundRectTemp.width > src.cols)
				foundRectTemp.width = src.cols - foundRectTemp.x;
			if (foundRectTemp.y + foundRectTemp.height > src.rows)
				foundRectTemp.height = src.rows - foundRectTemp.y;

			/*Save the box as hard example*/
			Mat HardExample = src(foundRectTemp);	//Hard example, crop from the box on the origin negative sample  
			resize(HardExample, HardExample, Size(64, 128));	//Resize to 64x128
			sprintf_s(saveName, "%shardexample%09d.jpg", _SaveDir, HardExDetectCount);
			imwrite(saveName, HardExample);

			/*Save hard example list*/
			sprintf_s(saveName, "hardexample%09d.jpg", HardExDetectCount);
			foutList << saveName;


			HardExDetectCount++;
			//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整   
			//rectangle(HardExTemp, foundRectTemp.tl(), foundRectTemp.br(), Scalar(0, 255, 0), 3);

		}
		std::cout << "                                                                    Detect: " << HardExDetectPer << endl;
		lineCount++;
	}
	std::cout << "Totally find " << HardExDetectCount << " hard example." << endl;

	return STATUS_OK;
}

int SVM_HOG_Training::HardExampleDetect(char* _SaveDir,
	vector<float> _Detector)
{
	Mat src;
	char saveName[1000];//剪裁出来的hard example图片的文件名  
	string ImgNameTemp;
	ifstream finNeg(NegListDir);					//Load negative sample list 
	int lineCount = 0;								//Line number for reading detector txt file
	int HardExDetectCount = 0;						//Number for counting total hard example

	/*Save hard example list*/
	char HardExListDir[1000];						//Directory of hard example list to save
	sprintf_s(HardExListDir, "%sHardExList.txt", _SaveDir);
	ofstream foutList(HardExListDir, ofstream::trunc);

	if (_Detector.size() != 3781)
	{
		cout << "The size of the detector is incorrect: " << _Detector.size() << endl;
		cout << "Please check your detector." << endl;
		return STATUS_ERROR;
	}

	HOGDescriptor hog;
	hog.setSVMDetector(_Detector);					//Set HOG detector  

	while (getline(finNeg, ImgNameTemp))
	{
		if (PRINT_DETAILLY) std::cout << "Detect HardExample, NegImg " << lineCount + 1 << "：" << ImgNameTemp << endl;
		ImgNameTemp = NegSampleDir + ImgNameTemp;
		src = imread(ImgNameTemp);
		if (!src.data)								// Check for invalid input
		{
			cout << "Could not open or find the image in NegImg " << lineCount + 1 << endl;
			cout << "Please check the diectory is incorrect." << endl;
			cout << ImgNameTemp << endl;
			return STATUS_ERROR;
		}

		int HardExDetectPer = 0;					//Count number of hard example in every negative sample

		Mat HardExTemp = src.clone();				//Avoid to modify origin negative sample
		vector<Rect> foundRect;						//Save all hard example box
		hog.detectMultiScale(src, foundRect, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		HardExDetectPer = foundRect.size();

		for (int i = 0; i < foundRect.size(); i++)
		{
			if (HardExDetectCount != 0)				//foutList line jump after the first line is saved
			{
				foutList << endl;
			}

			/*Force box boundary not be out of the boundary of NegSample*/
			Rect foundRectTemp = foundRect[i];
			if (foundRectTemp.x < 0)
				foundRectTemp.x = 0;
			if (foundRectTemp.y < 0)
				foundRectTemp.y = 0;
			if (foundRectTemp.x + foundRectTemp.width > src.cols)
				foundRectTemp.width = src.cols - foundRectTemp.x;
			if (foundRectTemp.y + foundRectTemp.height > src.rows)
				foundRectTemp.height = src.rows - foundRectTemp.y;

			/*Save the box as hard example*/
			Mat HardExample = src(foundRectTemp);	//Hard example, crop from the box on the origin negative sample  
			resize(HardExample, HardExample, Size(64, 128));	//Resize to 64x128
			sprintf_s(saveName, "%shardexample%09d.jpg", _SaveDir, HardExDetectCount);
			imwrite(saveName, HardExample);

			/*Save hard example list*/
			sprintf_s(saveName, "hardexample%09d.jpg", HardExDetectCount);
			foutList << saveName;

			HardExDetectCount++;
			//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整   
			//rectangle(HardExTemp, foundRectTemp.tl(), foundRectTemp.br(), Scalar(0, 255, 0), 3);
		}
		if (PRINT_DETAILLY) std::cout << "                                                                    Detect: " << HardExDetectPer << endl;
		lineCount++;
	}
	std::cout << "Totally find " << HardExDetectCount << " hard example." << endl;
	_HardExNumber_ = HardExDetectCount;

	return STATUS_OK;
}

int SVM_HOG_Training::HOGDetector(vector<float> _Detector,
	char* _SaveDir, vector<int> &_MissDetectorList)
{
	char SaveDir[256];
	char VideoOut[256];
	char DetectorDir[256];
	bool running;
	bool use_gpu;
	bool make_gray;
	double scale;
	double resize_scale;
	int win_width;
	int win_stride_width, win_stride_height;
	int gr_threshold;
	int nlevels;
	double hit_threshold;
	bool gamma_corr;

	string img_source;
	string vdo_source;
	string output;
	int camera_id;
	bool write_once;

	use_gpu = false;
	make_gray = 0;
	resize_scale = 1;
	win_width = 128;
	vdo_source = "VideoTest.mp4";
	img_source = "";
	output = "result.jpg";
	camera_id = -1;

	win_stride_width = 8;
	win_stride_height = 8;

	gr_threshold = 3;
	nlevels = 13;

	hit_threshold = 0.15;
	scale = 1.05;
	gamma_corr = true;
	write_once = true;
	/*-----------------------------------*/

	running = true;
	VideoWriter video_writer;
	if (DEBUG_MODE)
	{
		cout << "make_gray = " << make_gray << endl;
		cout << "resize_scale = " << resize_scale << endl;
		cout << "img_source = " << img_source << endl;
		cout << "output = " << output << endl;
		cout << "camera_id = " << camera_id << endl;
	}
	
	sprintf_s(VideoOut, sizeof(VideoOut), "%sVideoResult.avi", _SaveDir);
	Size win_size(64, 128);						//window size you want to detect

	cout << "video save in : " <<SaveDir << endl;

	Size win_stride(win_stride_width, win_stride_height);

	// Create HOG descriptors and detectors here

	/*ocl::HOGDescriptor gpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9,
		ocl::HOGDescriptor::DEFAULT_WIN_SIGMA, 0.2, gamma_corr,
		ocl::HOGDescriptor::DEFAULT_NLEVELS);*/
	HOGDescriptor cpu_hog(win_size, Size(16, 16), Size(8, 8), Size(8, 8), 9, 1, -1,
		HOGDescriptor::L2Hys, 0.2, gamma_corr, cv::HOGDescriptor::DEFAULT_NLEVELS);
	/*gpu_hog.setSVMDetector(_Detector);*/
	cpu_hog.setSVMDetector(_Detector);

	while (running)
	{
		VideoCapture vc;
		Mat frame;

		if (vdo_source != "")
		{
			vc.open(vdo_source.c_str());
			if (!vc.isOpened())
				throw runtime_error(string("can't open video file: " + vdo_source));
			vc >> frame;
		}
		else if (camera_id != -1)
		{
			vc.open(camera_id);
			if (!vc.isOpened())
			{
				stringstream msg;
				msg << "can't open camera: " << camera_id;
				throw runtime_error(msg.str());
			}
			vc >> frame;
		}
		else
		{
			frame = imread(img_source);
			if (frame.empty())
				throw runtime_error(string("can't open image file: " + img_source));
		}
		Mat img_aux, img, img_to_show;
		/*ocl::oclMat gpu_img;*/

		// Iterate over all frames
		bool verify = false;
		int MissDetectCount = 0;

		while (running && !frame.empty())
		{
			workBegin();
			// 			resize(frame, frame, Size(160, 120));
			resize(frame, frame, Size(320, 240));
			//			resize(frame, frame, Size(640, 480));
			// Change format of the image
			if (make_gray) cvtColor(frame, img_aux, CV_BGR2GRAY);
			else if (use_gpu) cvtColor(frame, img_aux, CV_BGR2BGRA);
			else frame.copyTo(img_aux);

			// Resize image
			if (abs(scale - 1.0) > 0.001)
			{
				Size sz((int)((double)img_aux.cols / resize_scale), (int)((double)img_aux.rows / resize_scale));
				resize(img_aux, img, sz);
			}
			else img = img_aux;
			img_to_show = img;
			/*gpu_hog.nlevels = nlevels;*/
			cpu_hog.nlevels = nlevels;
			vector<Rect> found;

			// Perform HOG classification
			hogWorkBegin();
			//if (use_gpu)
			//{
			//	gpu_img.upload(img);
			//	gpu_hog.detectMultiScale(gpu_img, found, hit_threshold, win_stride,
			//		Size(0, 0), scale, gr_threshold);
			//	if (!verify)
			//	{
			//		// verify if GPU output same objects with CPU at 1st run
			//		verify = true;
			//		vector<Rect> ref_rst;
			//		cvtColor(img, img, CV_BGRA2BGR);
			//		cpu_hog.detectMultiScale(img, ref_rst, hit_threshold, win_stride,
			//			Size(0, 0), scale, gr_threshold - 2);
			//		/*double accuracy = checkRectSimilarity(img.size(), ref_rst, found);
			//		cout << "\naccuracy value: " << accuracy << endl;*/
			//	}
			//}
			//else cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
			//	Size(0, 0), scale, gr_threshold);
			cpu_hog.detectMultiScale(img, found, hit_threshold, win_stride,
				Size(0, 0), scale, gr_threshold);

			//			else cpu_hog.detectMultiScale(img, found, 0, win_stride, Size(32, 32), 1.05, 2);
			hogWorkEnd();

			//count number of miss detection
			if (found.size() == 0)
			{
				MissDetectCount++;
			}

			// Draw positive classified windows
			for (size_t i = 0; i < found.size(); i++)
			{
				Rect r = found[i];
				rectangle(img_to_show, r.tl(), r.br(), CV_RGB(0, 255, 0), 3);
				//				cout << "r.tl = " << r.tl().x << "," << r.tl().y << "; r.br = " << r.br().x << "," << r.br().y << endl;
			}
			resize(img_to_show, img_to_show, Size(640, 480));
			if (use_gpu)
				putText(img_to_show, "Mode: GPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
			else
				putText(img_to_show, "Mode: CPU", Point(5, 25), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
			putText(img_to_show, "FPS (HOG only): " + hogWorkFps(), Point(5, 65), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
			putText(img_to_show, "FPS (total): " + workFps(), Point(5, 105), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
			if (DEBUG_MODE) imshow("opencv_gpu_hog", img_to_show);
			if (vdo_source != "" || camera_id != -1) vc >> frame;

			workEnd();

			if (VideoOut != "" && write_once)
			{
				if (img_source != "")     // wirte image
				{
					write_once = false;
					imwrite(output, img_to_show);
				}
				else                    //write video
				{
					if (!video_writer.isOpened())
					{
						video_writer.open(VideoOut, CV_FOURCC('x', 'v', 'i', 'd'), 24,
							img_to_show.size(), true);
						if (!video_writer.isOpened())
							throw std::runtime_error("can't create video writer");
					}

					if (make_gray) cvtColor(img_to_show, img, CV_GRAY2BGR);
					else cvtColor(img_to_show, img, CV_BGRA2BGR);
					video_writer << img;
				}
			}

		}
		if(DEBUG_MODE) cout << "miss count: " << MissDetectCount << " in all frame " << vc.get(CV_CAP_PROP_FRAME_COUNT) << endl;
		_MissDetectorList.push_back(MissDetectCount);
		break;
	}

}



inline void SVM_HOG_Training::hogWorkBegin()
{
	hog_work_begin = getTickCount();
}

inline void SVM_HOG_Training::hogWorkEnd()
{
	int64 delta = getTickCount() - hog_work_begin;
	double freq = getTickFrequency();
	hog_work_fps = freq / delta;
}

inline string SVM_HOG_Training::hogWorkFps() const
{
	stringstream ss;
	ss << hog_work_fps;
	return ss.str();
}

inline void SVM_HOG_Training::workBegin()
{
	work_begin = getTickCount();
}

inline void SVM_HOG_Training::workEnd()
{
	int64 delta = getTickCount() - work_begin;
	double freq = getTickFrequency();
	work_fps = freq / delta;
}

inline string SVM_HOG_Training::workFps() const
{
	stringstream ss;
	ss << work_fps;
	return ss.str();
}

vector<float> SVM_HOG_Training::getDetector()
{
	return DetectorTemp;
}

bool SVM_HOG_Training::dirExists(const char* dirName_in)
{
	DWORD ftyp = GetFileAttributesA(dirName_in);
	if (ftyp == INVALID_FILE_ATTRIBUTES)
		return false;								//something is wrong with your path!

	if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
		return true;								// this is a directory!

	return false;									// this is not a directory!
}

int SVM_HOG_Training::HardExampleIterationList(char* _SaveDir,
	int _FolderCount,
	int _HardExNumberTemp)
{
	char IterationListDir[1000];
	char HardExListDirTemp[1000];
	char HardExDirTemp[1000];
	char HardExDirTemp_Save[1000];
	string DataDirTemp;

	char SaveDir[1000];
	sprintf_s(SaveDir, sizeof(SaveDir), "%sHardExampleList.txt", _SaveDir);

	sprintf_s(HardExListDirTemp, sizeof(HardExListDirTemp), "%sHardExample%06d/HardExList.txt", _SaveDir, _FolderCount);
	sprintf_s(HardExDirTemp, sizeof(HardExDirTemp), "%sHardExample%06d/", _SaveDir, _FolderCount);
	sprintf_s(HardExDirTemp_Save, sizeof(HardExDirTemp_Save), "HardExample%06d/", _FolderCount);
	if (dirExists(HardExDirTemp))				//if the hard example iteration exist
	{
		ofstream fout(SaveDir, std::ios_base::app);
		ifstream fin(HardExListDirTemp);
		for (int num = 0; num<_HardExNumberTemp && getline(fin, DataDirTemp); num++)
		{
			DataDirTemp = string(HardExDirTemp_Save) + DataDirTemp;
			if (PRINT_DETAILLY) cout << DataDirTemp << endl;
			fout << DataDirTemp;
			fout << endl;
		}
	}

	return STATUS_OK;
}

int SVM_HOG_Training::HOG_SVM_Train_iteration(bool _IterationOrNot,
	int _IterationTimes,
	char* _SaveDir,
	char* _NegWindowsListDir,
	char* _NegWindowsDir,
	int _NegWindowsNumber)
{
	IterationCount = 0;
	int HardExCount = 0;
	int HardExampleIterationNumber = 0;					//MMMM
	IterationTimes = _IterationTimes;
	IterationOrNot = _IterationOrNot;
	bool TrainOrNot = true;
	bool UseHardExOrNot_First = false;
	bool UseHardExOrNot_Other = true;
	int HardExNumberTemp = 0;

	char HardExampleIterationListDir[1000];
	sprintf_s(HardExampleIterationListDir, sizeof(HardExampleIterationListDir), "%sHardExampleList.txt", _SaveDir);

	char* HardExampleIterationDir = _SaveDir;			//MMMM

	char HardExSaveDir[1000];
	vector<float> Detector;							//get the detector after training

	char HardExListDirTemp_char[1000];

	for (IterationCount; IterationCount < _IterationTimes + 1; IterationCount++)
	{
		cout << "IterationCount: " << IterationCount << endl;

		if (!IterationCount)					//a complete train with once hardexample
		{
			/*HardExNumberTemp, HardExListDirTemp and HardExDirTemp doesn't matter for the first times*/
			HOG_SVM_Train(TrainOrNot, HardExampleIterationNumber, string(HardExampleIterationListDir),
				string(HardExampleIterationDir), UseHardExOrNot_First, _SaveDir,
				_NegWindowsListDir, _NegWindowsDir, _NegWindowsNumber);
			if (DEBUG_MODE) system("pause");

			Detector = getDetector();

			if (PRINT_DETAILLY)
			{
				for (size_t i = 0; i < Detector.size(); i++)
				{
					std::cout << Detector[i] << endl;
				}
			}

			if (DEBUG_MODE) system("pause");

			sprintf_s(HardExSaveDir, sizeof(HardExSaveDir), "%sHardExample%06d/", _SaveDir, HardExCount);

			if (!dirExists(HardExSaveDir))			//if the directory is not exist, create new one
			{
				cout << "We'll create a folder, " << HardExSaveDir << ", to save hardexample with iteration"
					<< IterationCount << endl;
				if (DEBUG_MODE) system("pause");
				CreateDirectory(HardExSaveDir, NULL);
			}

			HardExampleDetect(HardExSaveDir, Detector);
			HardExampleIterationList(_SaveDir, HardExCount, _HardExNumber_);
			HardExampleIterationNumber = HardExampleIterationNumber + _HardExNumber_;

			HardExCount++;

			if (DEBUG_MODE) system("pause");

			/*First hard example train*/
			HOG_SVM_Train(TrainOrNot, HardExampleIterationNumber, string(HardExampleIterationListDir),
				string(HardExampleIterationDir), UseHardExOrNot_Other, _SaveDir,
				_NegWindowsListDir, _NegWindowsDir, _NegWindowsNumber);
			if (DEBUG_MODE) system("pause");

			Detector = getDetector();

			if (PRINT_DETAILLY)
			{
				for (size_t i = 0; i < Detector.size(); i++)
				{
					std::cout << Detector[i] << endl;
				}
			}
			if (DEBUG_MODE) system("pause");

			sprintf_s(HardExSaveDir, sizeof(HardExSaveDir), "%sHardExample%06d/", _SaveDir, HardExCount);
			if (!dirExists(HardExSaveDir))			//if the directory is not exist, create new one
			{
				cout << "We'll create a folder, " << HardExSaveDir << ", to save hardexample with iteration"
					<< IterationCount << endl;
				CreateDirectory(HardExSaveDir, NULL);
			}

			HardExampleDetect(HardExSaveDir, Detector);
			HardExampleIterationList(_SaveDir, HardExCount, _HardExNumber_);
			HardExampleIterationNumber = HardExampleIterationNumber + _HardExNumber_;

			HardExCount++;

			//HOGDetector(Detector, HardExSaveDir);

			if (DEBUG_MODE) system("pause");

		}
		else
		{

			/*Last hard example will put in to train*/
			HOG_SVM_Train(TrainOrNot, HardExampleIterationNumber, string(HardExampleIterationListDir),
				string(HardExampleIterationDir), UseHardExOrNot_Other, _SaveDir,
				_NegWindowsListDir, _NegWindowsDir, _NegWindowsNumber);
			if (DEBUG_MODE) system("pause");

			Detector = getDetector();
			if (PRINT_DETAILLY)
			{
				for (size_t i = 0; i < Detector.size(); i++)
				{
					std::cout << Detector[i] << endl;
				}
			}
			if (DEBUG_MODE) system("pause");

			sprintf_s(HardExSaveDir, sizeof(HardExSaveDir), "%sHardExample%06d/", _SaveDir, HardExCount);
			if (!dirExists(HardExSaveDir))			//if the directory is not exist, create new one
			{
				cout << "We'll create a folder, " << HardExSaveDir << ", to save hardexample with iteration"
					<< IterationCount << endl;
				CreateDirectory(HardExSaveDir, NULL);
			}

			HardExampleDetect(HardExSaveDir, Detector);
			HardExampleIterationList(_SaveDir, HardExCount, _HardExNumber_);
			HardExampleIterationNumber = HardExampleIterationNumber + _HardExNumber_;

			HardExCount++;

			//HOGDetector(Detector, HardExSaveDir);

			if (DEBUG_MODE) system("pause");
		}

	}

	return STATUS_OK;
}

cv::Scalar SVM_HOG_Training::calMSSIM(const Mat& _MatrixReference, const Mat& _MatrixUnderTest)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	Mat MatrixReference, MatrixUnderTest;
	_MatrixReference.convertTo(MatrixReference, d);           // cannot calculate on one byte large values
	_MatrixUnderTest.convertTo(MatrixUnderTest, d);

	Mat MatrixUnderTest_sqrt = MatrixUnderTest.mul(MatrixUnderTest);        // MatrixUnderTest^2
	Mat MatrixReference_sqrt = MatrixReference.mul(MatrixReference);        // MatrixReference^2
	Mat MatrixR_MatrixU_multi = MatrixReference.mul(MatrixUnderTest);        // MatrixReference * MatrixUnderTest

	/***********************PRELIMINARY COMPUTING ******************************/

	Mat MatrixReference_Gaussian, MatrixUnderTest_Gaussion;
	GaussianBlur(MatrixReference, MatrixReference_Gaussian, Size(11, 11), 1.5);
	GaussianBlur(MatrixUnderTest, MatrixUnderTest_Gaussion, Size(11, 11), 1.5);

	Mat MatrixReference_Gaussion_sqrt = MatrixReference_Gaussian.mul(MatrixReference_Gaussian);
	Mat MatrixUnderTest_Gaussion_sqrt = MatrixUnderTest_Gaussion.mul(MatrixUnderTest_Gaussion);
	Mat MatrixR_MatrixU_Gaussion_multi = MatrixReference_Gaussian.mul(MatrixUnderTest_Gaussion);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(MatrixUnderTest_sqrt, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= MatrixReference_Gaussion_sqrt;

	GaussianBlur(MatrixReference_sqrt, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= MatrixUnderTest_Gaussion_sqrt;

	GaussianBlur(MatrixR_MatrixU_multi, sigma12, Size(11, 11), 1.5);
	sigma12 -= MatrixR_MatrixU_Gaussion_multi;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * MatrixR_MatrixU_Gaussion_multi + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = MatrixReference_Gaussion_sqrt + MatrixUnderTest_Gaussion_sqrt + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
	return mssim;
}

double SVM_HOG_Training::calPSNR(const Mat& _MatrixReference, const Mat& _MatrixUnderTest)
{
	Mat _AbsDiff;
	absdiff(_MatrixReference, _MatrixUnderTest, _AbsDiff);      // |MatrixReference - MatrixUnderTest|
	_AbsDiff.convertTo(_AbsDiff, CV_32F);						// cannot make a square on 8 bits
	_AbsDiff = _AbsDiff.mul(_AbsDiff);							// |MatrixReference - MatrixUnderTest|^2

	Scalar _Sum = sum(_AbsDiff);								// sum elements per channel

	double sse = _Sum.val[0] + _Sum.val[1] + _Sum.val[2];		// sum channels

	if (sse <= 1e-10)											// for small values return zero
		return 0;
	else
	{
		double  mse = sse / (double)(_MatrixReference.channels() * _MatrixReference.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}

int SVM_HOG_Training::PSNR_SSIMTestDemo()
{

	char* TestListDir = "PSNRTest/TestList.txt";
	char* TestImageDir = "PSNRTest/";
	string TestImage = "";
	ifstream fin(TestListDir);
	int count = 0;
	vector<cv::Mat> Image;
	Mat ImageTemp;
	vector<cv::Scalar> SSIM;
	cv::Scalar SSIMTemp;
	vector<double> PSNR;
	double PSNRTemp;

	while (getline(fin, TestImage))
	{
		count++;
		TestImage = TestImageDir + TestImage;
		ImageTemp = imread(TestImage);
		if (!ImageTemp.data)
		{
			cout << "can't open the image" << endl;
			cout << "Please check the directory: " << TestImage << endl;
			system("pause");
		}
		Image.push_back(ImageTemp);
		imshow("Img", ImageTemp);
		waitKey(30);
	}
	cout << " Test, Image in PSNRTest/" << endl;

	for (size_t i = 0; i <Image.size() / 2; i++)
	{
		SSIMTemp = calMSSIM(Image[2 * i], Image[2 * i + 1]);
		SSIM.push_back(SSIMTemp);
		PSNRTemp = calPSNR(Image[2 * i], Image[2 * i + 1]);
		PSNR.push_back(PSNRTemp);
		cout << "SSIM: " << SSIMTemp << " of " << 2 * i + 1 << " " << 2 * i + 2 << endl;
		cout << "PSNR: " << PSNRTemp << " of " << 2 * i + 1 << " " << 2 * i + 2 << endl;
		cout << "/--------------------------------/" << endl;
	}

	return STATUS_OK;
}

/*
*http://docs.opencv.org/2.4/doc/tutorials/features2d/feature_flann_matcher/feature_flann_matcher.html
*/
int SVM_HOG_Training::FeatureMatchingFLANN(cv::Mat _InputImage1,
	cv::Mat _InputImage2)
{
	/*Mat InputImage1_Gray;
	cvtColor(_InputImage1, InputImage1_Gray, CV_BGR2GRAY);
	Mat InputImage2_Gray;
	cvtColor(_InputImage2, InputImage2_Gray, CV_BGR2GRAY);*/

	Mat InputImage1_Gray = _InputImage1;
	Mat InputImage2_Gray = _InputImage2;

	if (!InputImage1_Gray.data || !InputImage2_Gray.data)
	{
		cout << "Error image input" << endl;
		return STATUS_ERROR;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;

	Ptr <xfeatures2d::SURF> detector = xfeatures2d::SURF::create(minHessian);

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector->detect(InputImage1_Gray, keypoints_1);
	if (keypoints_1.size() == 0)
	{
		imshow("Error", _InputImage1);
		waitKey(30);
		cout << "Can't find feature with the input image" << endl;
		return STATUS_ERROR;
	}

	detector->detect(InputImage2_Gray, keypoints_2);
	if (keypoints_2.size() == 0)
	{
		imshow("Error", _InputImage1);
		waitKey(30);
		cout << "Can't find feature with the input image" << endl;
		return STATUS_ERROR;
	}

	//-- Step 2: Calculate descriptors (feature vectors)

	Mat descriptors_1, descriptors_2;

	detector->compute(InputImage1_Gray, keypoints_1, descriptors_1);
	detector->compute(InputImage2_Gray, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	if (PRINT_DETAILLY) printf("-- Max dist : %f \n", max_dist);
	if (PRINT_DETAILLY) printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	if (DEBUG_MODE)
	{
		Mat img_matches;
		drawMatches(InputImage1_Gray, keypoints_1, InputImage2_Gray, keypoints_2,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		//-- Show detected matches
		imshow("Good Matches", img_matches);

		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
		}

		waitKey(0);
	}

	return good_matches.size();
}

int SVM_HOG_Training::FLANNDemo()
{
	char* TestListDir = "PSNRTest/TestList.txt";
	char* TestImageDir = "PSNRTest/";
	string TestImage = "";
	ifstream fin(TestListDir);
	int count = 0;
	vector<cv::Mat> Image;
	Mat ImageTemp;
	vector<cv::Scalar> SSIM;
	cv::Scalar SSIMTemp;
	vector<double> PSNR;
	double PSNRTemp;

	while (getline(fin, TestImage))
	{
		count++;
		TestImage = TestImageDir + TestImage;
		ImageTemp = imread(TestImage);
		if (!ImageTemp.data)
		{
			cout << "can't open the image" << endl;
			cout << "Please check the directory: " << TestImage << endl;
			system("pause");
		}
		Image.push_back(ImageTemp);
		imshow("Img", ImageTemp);
		waitKey(30);
	}
	cout << " Test, Image in PSNRTest/" << endl;

	for (size_t i = 0; i <Image.size() / 2; i++)
	{

		cout << "Match: " << FeatureMatchingFLANN(Image[2 * i], Image[2 * i + 1]) << endl;
		cout << "/--------------------------------/" << endl;
	}

}

/*Use DAISY descriptor and SSIM to find similar structure of two input image*/
int SVM_HOG_Training::FeatureHomography(cv::Mat _InputImage1,
	cv::Mat _InputImage2)
{
	int ErrorCount = 0;

	if (!_InputImage1.data || !_InputImage2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	const float nn_match_ratio = 0.7f;
	const float keypoint_diameter = 15.0f;

	// Load images
	Mat img_object;
	cvtColor(_InputImage1, img_object, CV_BGR2GRAY);
	Mat img_scene;
	cvtColor(_InputImage2, img_scene, CV_BGR2GRAY);

	vector<KeyPoint> keypoints_object, keypoints_scene;

	// Add every pixel to the list of keypoints for each image

	cv::Ptr<xfeatures2d::SIFT> SIFTDetector = xfeatures2d::SIFT::create();
	//Ptr<cv::xfeatures2d::SURF> SURFDetector = xfeatures2d::SURF::create(400);

	SIFTDetector->detect(img_object, keypoints_object);
	//SURFDetector->detect(img_object, keypoints_object);
	if (keypoints_object.size() == 0)
	{
		imshow("Error", _InputImage1);
		waitKey(30);
		cout << "Can't find feature with the input image" << endl;
		ErrorCount++;
	}

	SIFTDetector->detect(img_scene, keypoints_scene);
	//SURFDetector->detect(img_scene, keypoints_scene);
	if (keypoints_scene.size() == 0)
	{
		imshow("Error", _InputImage2);
		waitKey(30);
		cout << "Can't find feature with the input image" << endl;
		ErrorCount++;
	}

	if (ErrorCount > 0)
	{
		cout << "Keypoint1: " << keypoints_object.size() << ", Keypoint2: " << keypoints_scene.size() << endl;
		return STATUS_ERROR;
	}

	Mat descriptors_object, descriptors_scene;

	
	Ptr<cv::xfeatures2d::DAISY> descriptor_extractor = cv::xfeatures2d::DAISY::create();

	// Compute DAISY descriptors for both images 

	descriptor_extractor->compute(img_object, keypoints_object, descriptors_object);
	descriptor_extractor->compute(img_scene, keypoints_scene, descriptors_scene);

	/*SIFTDetector->compute(img_object, keypoints_object, descriptors_object);
	SIFTDetector->compute(img_scene, keypoints_scene, descriptors_scene);*/
	/*SURFDetector->compute(img_object, keypoints_object, descriptors_object);
	SURFDetector->compute(img_scene, keypoints_scene, descriptors_scene);*/

	vector <vector<DMatch>> matches;

	// For each descriptor in image1, find 2 closest matched in image2 (note: couldn't get BF matcher to work here at all)
	FlannBasedMatcher flannmatcher;
	flannmatcher.add(descriptors_object);
	flannmatcher.train();
	flannmatcher.knnMatch(descriptors_scene, matches, 5);


	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints_object[first.trainIdx]);
			matched2.push_back(keypoints_scene[first.queryIdx]);
			cout << "left point:" << keypoints_object[first.trainIdx].pt << ", right point:" << keypoints_scene[first.queryIdx].pt << endl;
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
	}

	if (good_matches.size()<4)
	{
		cout << "Not Enough matches!" << endl;
		return STATUS_ERROR;
	}

	Mat img_matches_1;
	drawMatches(img_object, matched1, img_scene, matched2, good_matches, img_matches_1);

	imshow("123", img_matches_1);
	waitKey(30);

	//-- Localize the object
	std::vector<Point2f> obj; 
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(matched1[i].pt);
		scene.push_back(matched2[i].pt);
		if (DEBUG_MODE) cout << "Left point Keypoint " << keypoints_object[good_matches[i].queryIdx].pt << ", Right point Keypoint " << keypoints_scene[good_matches[i].trainIdx].pt << endl;
	}

	cv::Mat H = findHomography(obj, scene, RANSAC);
	
	if (!H.data)
	{
		cout << "Homo not find" << endl;
		return STATUS_ERROR;
	}

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	if (DEBUG_MODE)
	{
		for (int i = 0; i < scene_corners.size(); i++)
		{
			cout << "ori corner " << i << ": " << obj_corners[i] << endl;
			cout << "corner " << i << ": " << scene_corners[i] << endl;
		}
	}

	cv::Mat Response_obj, Response_scene;

	int x_temp_obj = 0, x_temp_scene = 0, y_temp_obj = 0, y_temp_scene = 0;
	Size sizeTemp(64 - abs(cvRound(scene_corners[0].x)), 128 - abs(cvRound(scene_corners[0].y)));
	if ((sizeTemp.width<0) || (sizeTemp.height<0))
	{
		if (PRINT_DETAILLY) cout << "It warp" << endl;
		return STATUS_ERROR;
	}

	if (cvRound(scene_corners[0].x)<0)
	{
		x_temp_obj = x_temp_obj - cvRound(scene_corners[0].x);
	}
	else
	{
		x_temp_scene = x_temp_scene + cvRound(scene_corners[0].x);
	}

	if (cvRound(scene_corners[0].y)<0)
	{
		y_temp_obj = y_temp_obj - cvRound(scene_corners[0].y);
	}
	else
	{
		y_temp_scene = y_temp_scene + cvRound(scene_corners[0].y);
	}
	Point upLeftCorner_obj(x_temp_obj, y_temp_obj),
		upLeftCorner_scene(x_temp_scene, y_temp_scene);

	Response_obj = _InputImage1(Rect(upLeftCorner_obj, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}
	if (DEBUG_MODE) imshow("Response obj", Response_obj);



	Response_scene = _InputImage2(Rect(upLeftCorner_scene, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}


	if (DEBUG_MODE) imshow("Response scene", Response_scene);

	

	if (DEBUG_MODE)
	{
		Mat img_matches_2 = img_matches_1.clone();

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches_2);

		waitKey(30);
	}

	waitKey(0);
	return 0;
}

std::vector < cv::KeyPoint> SVM_HOG_Training::KeyPointTest_SIFT(cv::Mat _InputImage)
{
	Mat InputImage_Gray;
	cvtColor(_InputImage, InputImage_Gray, CV_BGR2GRAY);

	if (!InputImage_Gray.data || !InputImage_Gray.data)
	{
		cout << "Error image input" << endl;
	}

	//-- Step 1: Detect the keypoints using SIFT Detector

	cv::Ptr<xfeatures2d::SIFT> SIFTDetector = xfeatures2d::SIFT::create(0, 3, 0.025, 10, 1.6);

	std::vector<KeyPoint> keypoints;

	SIFTDetector->detect(InputImage_Gray, keypoints);
	if (keypoints.size() == 0)
	{
		cout << "Can't find feature with the input image" << endl;
	}
	else
	{
		if (DEBUG_MODE)
		{
			Mat output;
			drawKeypoints(InputImage_Gray, keypoints, output);
			imshow("Keypoint", output);
			waitKey(30);

			cout << "Find Keypoint: " << keypoints.size() << endl;
		}

	}

	return keypoints;
}

int SVM_HOG_Training::HomographyTest_SIFT_DAISY(cv::Mat _InputImage1,
	cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2)
{
	//vector<cv::Mat> outHomographyImage;
	Mat img_object;
	cvtColor(_InputImage1, img_object, CV_BGR2GRAY);
	Mat img_scene;
	cvtColor(_InputImage2, img_scene, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<xfeatures2d::SIFT> SIFTDetector = xfeatures2d::SIFT::create(0, 3, 0.025, 10, 1.6);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	SIFTDetector->detect(img_object, keypoints_object);
	SIFTDetector->detect(img_scene, keypoints_scene);

	if (DEBUG_MODE) cout << "keypoint_obj: " << keypoints_object.size() << "keypoint_scene" << keypoints_scene.size() << endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	Ptr<cv::xfeatures2d::DAISY> DAISYDescriptor = cv::xfeatures2d::DAISY::create();

	DAISYDescriptor->compute(img_object, keypoints_object, descriptors_object);
	DAISYDescriptor->compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	vector <vector<DMatch>> matches;

	FlannBasedMatcher flannmatcher;
	flannmatcher.add(descriptors_object);
	flannmatcher.train();
	flannmatcher.knnMatch(descriptors_scene, matches, 5);

	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;
	const float nn_match_ratio = 0.7f;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints_object[first.trainIdx]);
			matched2.push_back(keypoints_scene[first.queryIdx]);
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}



	if (DEBUG_MODE)
	{
		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
		}
	}
	if (good_matches.size()<4)
	{
		cout << "Not Enough matches!" << endl;
		return STATUS_ERROR;
	}

	cv::Mat img_matches_1;
	if (DEBUG_MODE)
	{
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches_1);
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(matched1[i].pt);
		scene.push_back(matched2[i].pt);
		if (DEBUG_MODE) cout << "Left point Keypoint " << matched1[i].pt << ", Right point Keypoint " << matched2[i].pt << endl;
	}

	cv::Mat H = findHomography(obj, scene, RANSAC);
	if (!H.data)
	{
		cout << "Can't find Homography" << endl;
		return STATUS_ERROR;
	}

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	if (DEBUG_MODE)
	{
		for (int i = 0; i < scene_corners.size(); i++)
		{
			cout << "ori corner " << i << ": " << obj_corners[i] << endl;
			cout << "corner " << i << ": " << scene_corners[i] << endl;
		}
	}

	cv::Mat Response_obj, Response_scene;

	int x_temp_obj = 0, x_temp_scene = 0, y_temp_obj = 0, y_temp_scene = 0;
	Size sizeTemp(64 - abs(cvRound(scene_corners[0].x)), 128 - abs(cvRound(scene_corners[0].y)));
	if ((sizeTemp.width<0) || (sizeTemp.height<0))
	{
		if (PRINT_DETAILLY) cout << "It warp" << endl;
		return STATUS_ERROR;
	}

	if (cvRound(scene_corners[0].x)<0)
	{
		x_temp_obj = x_temp_obj - cvRound(scene_corners[0].x);
	}
	else
	{
		x_temp_scene = x_temp_scene + cvRound(scene_corners[0].x);
	}

	if (cvRound(scene_corners[0].y)<0)
	{
		y_temp_obj = y_temp_obj - cvRound(scene_corners[0].y);
	}
	else
	{
		y_temp_scene = y_temp_scene + cvRound(scene_corners[0].y);
	}
	Point upLeftCorner_obj(x_temp_obj, y_temp_obj),
		upLeftCorner_scene(x_temp_scene, y_temp_scene);



	Response_obj = _InputImage1(Rect(upLeftCorner_obj, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}
	if (DEBUG_MODE) imshow("Response obj", Response_obj);



	Response_scene = _InputImage2(Rect(upLeftCorner_scene, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}

	if (DEBUG_MODE) imshow("Response scene", Response_scene);

	if (DEBUG_MODE)
	{
		Mat img_matches_2 = img_matches_1.clone();

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches_2);

		waitKey(30);
	}

	Response_obj.copyTo(_OutImage1);
	Response_scene.copyTo(_OutImage2);

	return STATUS_OK;
}

int SVM_HOG_Training::HomographyTest_SURF_SURF(cv::Mat _InputImage1,
	cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2)
{
	//vector<cv::Mat> outHomographyImage;
	Mat img_object;
	cvtColor(_InputImage1, img_object, CV_BGR2GRAY);
	Mat img_scene;
	cvtColor(_InputImage2, img_scene, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<xfeatures2d::SURF> SURFDetector = xfeatures2d::SURF::create(200, 4, 3, false, false);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	SURFDetector->detect(img_object, keypoints_object);
	SURFDetector->detect(img_scene, keypoints_scene);

	if (DEBUG_MODE) cout << "keypoint_obj: " << keypoints_object.size() << "keypoint_scene" << keypoints_scene.size() << endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;

	SURFDetector->compute(img_object, keypoints_object, descriptors_object);
	SURFDetector->compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	vector <vector<DMatch>> matches;

	FlannBasedMatcher flannmatcher;
	flannmatcher.add(descriptors_object);
	flannmatcher.train();
	flannmatcher.knnMatch(descriptors_scene, matches, 5);

	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;
	const float nn_match_ratio = 0.7f;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints_object[first.trainIdx]);
			matched2.push_back(keypoints_scene[first.queryIdx]);
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}

	if (DEBUG_MODE)
	{
		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
		}
	}
	if (good_matches.size()<4)
	{
		cout << "Not Enough matches!" << endl;
		return STATUS_ERROR;
	}

	cv::Mat img_matches_1;
	if (DEBUG_MODE)
	{
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches_1);
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(matched1[i].pt);
		scene.push_back(matched2[i].pt);
		if (DEBUG_MODE) cout << "Left point Keypoint " << matched1[i].pt << ", Right point Keypoint " << matched2[i].pt << endl;
	}

	cv::Mat H = findHomography(obj, scene, RANSAC);
	if (!H.data)
	{
		cout << "Can't find Homography" << endl;
		return STATUS_ERROR;
	}

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	if (DEBUG_MODE)
	{
		for (int i = 0; i < scene_corners.size(); i++)
		{
			cout << "ori corner " << i << ": " << obj_corners[i] << endl;
			cout << "corner " << i << ": " << scene_corners[i] << endl;
		}
	}

	cv::Mat Response_obj, Response_scene;

	int x_temp_obj = 0, x_temp_scene = 0, y_temp_obj = 0, y_temp_scene = 0;
	Size sizeTemp(64 - abs(cvRound(scene_corners[0].x)), 128 - abs(cvRound(scene_corners[0].y)));
	if ((sizeTemp.width<0) || (sizeTemp.height<0))
	{
		if (PRINT_DETAILLY) cout << "It warp" << endl;
		return STATUS_ERROR;
	}

	if (cvRound(scene_corners[0].x)<0)
	{
		x_temp_obj = x_temp_obj - cvRound(scene_corners[0].x);
	}
	else
	{
		x_temp_scene = x_temp_scene + cvRound(scene_corners[0].x);
	}

	if (cvRound(scene_corners[0].y)<0)
	{
		y_temp_obj = y_temp_obj - cvRound(scene_corners[0].y);
	}
	else
	{
		y_temp_scene = y_temp_scene + cvRound(scene_corners[0].y);
	}
	Point upLeftCorner_obj(x_temp_obj, y_temp_obj),
		upLeftCorner_scene(x_temp_scene, y_temp_scene);



	Response_obj = _InputImage1(Rect(upLeftCorner_obj, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}
	if (DEBUG_MODE) imshow("Response obj", Response_obj);



	Response_scene = _InputImage2(Rect(upLeftCorner_scene, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}

	if (DEBUG_MODE) imshow("Response scene", Response_scene);

	if (DEBUG_MODE)
	{
		Mat img_matches_2 = img_matches_1.clone();

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches_2);

		waitKey(30);
	}

	Response_obj.copyTo(_OutImage1);
	Response_scene.copyTo(_OutImage2);

	return STATUS_OK;
}

int SVM_HOG_Training::HomographyTest_SURF_SIFT(cv::Mat _InputImage1,
	cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2)
{
	//vector<cv::Mat> outHomographyImage;
	Mat img_object;
	cvtColor(_InputImage1, img_object, CV_BGR2GRAY);
	Mat img_scene;
	cvtColor(_InputImage2, img_scene, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<xfeatures2d::SURF> SURFDetector = xfeatures2d::SURF::create(200, 4, 3, false, false);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	SURFDetector->detect(img_object, keypoints_object);
	SURFDetector->detect(img_scene, keypoints_scene);

	if (DEBUG_MODE) cout << "keypoint_obj: " << keypoints_object.size() << "keypoint_scene" << keypoints_scene.size() << endl;

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	Ptr<xfeatures2d::SIFT> SIFTDescripter = xfeatures2d::SIFT::create();

	SIFTDescripter->compute(img_object, keypoints_object, descriptors_object);
	SIFTDescripter->compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	vector <vector<DMatch>> matches;

	FlannBasedMatcher flannmatcher;
	flannmatcher.add(descriptors_object);
	flannmatcher.train();
	flannmatcher.knnMatch(descriptors_scene, matches, 5);

	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;
	const float nn_match_ratio = 0.7f;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints_object[first.trainIdx]);
			matched2.push_back(keypoints_scene[first.queryIdx]);
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}

	if (DEBUG_MODE)
	{
		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
		}
	}
	if (good_matches.size()<4)
	{
		cout << "Not Enough matches!" << endl;
		return STATUS_ERROR;
	}

	cv::Mat img_matches_1;
	if (DEBUG_MODE)
	{
		drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches_1);
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(matched1[i].pt);
		scene.push_back(matched2[i].pt);
		if (DEBUG_MODE) cout << "Left point Keypoint " << matched1[i].pt << ", Right point Keypoint " << matched2[i].pt << endl;
	}

	cv::Mat H = findHomography(obj, scene, RANSAC);
	if (!H.data)
	{
		cout << "Can't find Homography" << endl;
		return STATUS_ERROR;
	}

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	if (DEBUG_MODE)
	{
		for (int i = 0; i < scene_corners.size(); i++)
		{
			cout << "ori corner " << i << ": " << obj_corners[i] << endl;
			cout << "corner " << i << ": " << scene_corners[i] << endl;
		}
	}

	cv::Mat Response_obj, Response_scene;

	int x_temp_obj = 0, x_temp_scene = 0, y_temp_obj = 0, y_temp_scene = 0;
	Size sizeTemp(64 - abs(cvRound(scene_corners[0].x)), 128 - abs(cvRound(scene_corners[0].y)));
	if ((sizeTemp.width<0) || (sizeTemp.height<0))
	{
		if (PRINT_DETAILLY) cout << "It warp" << endl;
		return STATUS_ERROR;
	}

	if (cvRound(scene_corners[0].x)<0)
	{
		x_temp_obj = x_temp_obj - cvRound(scene_corners[0].x);
	}
	else
	{
		x_temp_scene = x_temp_scene + cvRound(scene_corners[0].x);
	}

	if (cvRound(scene_corners[0].y)<0)
	{
		y_temp_obj = y_temp_obj - cvRound(scene_corners[0].y);
	}
	else
	{
		y_temp_scene = y_temp_scene + cvRound(scene_corners[0].y);
	}
	Point upLeftCorner_obj(x_temp_obj, y_temp_obj),
		upLeftCorner_scene(x_temp_scene, y_temp_scene);



	Response_obj = _InputImage1(Rect(upLeftCorner_obj, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}
	if (DEBUG_MODE) imshow("Response obj", Response_obj);



	Response_scene = _InputImage2(Rect(upLeftCorner_scene, sizeTemp));
	if (!Response_obj.data)
	{
		cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}

	if (DEBUG_MODE) imshow("Response scene", Response_scene);

	if (DEBUG_MODE)
	{
		Mat img_matches_2 = img_matches_1.clone();

		//-- Draw lines between the corners (the mapped object in the scene - image_2 )
		line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
		line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

		//-- Show detected matches
		imshow("Good Matches & Object detection", img_matches_2);

		waitKey(30);
	}

	Response_obj.copyTo(_OutImage1);
	Response_scene.copyTo(_OutImage2);

	return STATUS_OK;
}

int SVM_HOG_Training::HomographyTest_SIFT_SURF(cv::Mat _InputImage1,
	cv::Mat _InputImage2, cv::Mat &_OutImage1, cv::Mat &_OutImage2, std::vector<int> &_BoxSize)
{
	destroyAllWindows();

	//vector<cv::Mat> outHomographyImage;
	Mat img_object;
	cvtColor(_InputImage1, img_object, CV_BGR2GRAY);
	Mat img_scene;
	cvtColor(_InputImage2, img_scene, CV_BGR2GRAY);

	//-- Step 1: Detect the keypoints using SURF Detector
	Ptr<xfeatures2d::SIFT> SIFTDetector = xfeatures2d::SIFT::create(0, 3, 0.025, 10, 1.6);

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	SIFTDetector->detect(img_object, keypoints_object);
	SIFTDetector->detect(img_scene, keypoints_scene);

	if (DEBUG_MODE) cout << "keypoint_obj: " << keypoints_object.size() << "keypoint_scene" << keypoints_scene.size() << endl;

	if ((!keypoints_object.size()) || (!keypoints_scene.size()))
	{
		if (!IGNORE) cout << "Can't find keypoint" << endl;
		return STATUS_ERROR;
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	Mat descriptors_object, descriptors_scene;
	Ptr<xfeatures2d::SURF> SURFDescripter = xfeatures2d::SURF::create();

	SURFDescripter->compute(img_object, keypoints_object, descriptors_object);
	SURFDescripter->compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	vector <vector<DMatch>> matches;

	FlannBasedMatcher flannmatcher;
	flannmatcher.add(descriptors_object);
	flannmatcher.train();
	flannmatcher.knnMatch(descriptors_scene, matches, 5);

	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;
	const float nn_match_ratio = 0.7f;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints_object[first.trainIdx]);
			matched2.push_back(keypoints_scene[first.queryIdx]);
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}

	if (DEBUG_MODE)
	{
		for (int i = 0; i < (int)good_matches.size(); i++)
		{
			printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  distance : %f \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx, good_matches[i].distance);
		}
	}
	if (good_matches.size()<4)
	{
		if (!IGNORE) cout << "Not Enough matches!" << endl;
		return STATUS_ERROR;
	}

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj.push_back(matched1[i].pt);
		scene.push_back(matched2[i].pt);
		if (DEBUG_MODE) cout << "Left point Keypoint " << matched1[i].pt << ", Right point Keypoint " << matched2[i].pt << endl;
	}

	cv::Mat img_goodmatches, img_matches;
	if (DEBUG_MODE)
	{
		if ((good_matches.size()<keypoints_object.size()) && (good_matches.size()<keypoints_scene.size()))
		{
			drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
				good_matches, img_goodmatches);
			imshow("GoodMatch", img_goodmatches);
			waitKey(30);
		}
		else
		{
			if ((matches.size() < keypoints_object.size()) && (matches.size() < keypoints_scene.size()))
			{

				drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
					matches, img_matches);
				imshow("AllMatch", img_matches);
				waitKey(30);
			}

			if (!IGNORE) cout << "Can't draw good_match image because of matching fail" << endl;
		}

	}



	cv::Mat H = findHomography(obj, scene, RANSAC);
	if (!H.data)
	{
		if (!IGNORE) cout << "Can't find Homography" << endl;
		return STATUS_ERROR;
	}

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_object.cols, 0);
	obj_corners[2] = cvPoint(img_object.cols, img_object.rows); obj_corners[3] = cvPoint(0, img_object.rows);
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform(obj_corners, scene_corners, H);
	if (DEBUG_MODE)
	{
		for (int i = 0; i < scene_corners.size(); i++)
		{
			cout << "ori corner " << i << ": " << obj_corners[i] << endl;
			cout << "corner " << i << ": " << scene_corners[i] << endl;
		}
	}

	cv::Mat Response_obj, Response_scene;

	int x_temp_obj = 0, x_temp_scene = 0, y_temp_obj = 0, y_temp_scene = 0;
	Size sizeTemp(64 - abs(cvRound(scene_corners[0].x)), 128 - abs(cvRound(scene_corners[0].y)));
	if ((sizeTemp.width<0) || (sizeTemp.height<0))
	{
		if (PRINT_DETAILLY) cout << "It warp" << endl;
		return STATUS_ERROR;
	}


	if (cvRound(scene_corners[0].x)<0)
	{
		x_temp_obj = x_temp_obj - cvRound(scene_corners[0].x);
	}
	else
	{
		x_temp_scene = x_temp_scene + cvRound(scene_corners[0].x);
	}

	if (cvRound(scene_corners[0].y)<0)
	{
		y_temp_obj = y_temp_obj - cvRound(scene_corners[0].y);
	}
	else
	{
		y_temp_scene = y_temp_scene + cvRound(scene_corners[0].y);
	}
	Point upLeftCorner_obj(x_temp_obj, y_temp_obj),
		upLeftCorner_scene(x_temp_scene, y_temp_scene);



	Response_obj = _InputImage1(Rect(upLeftCorner_obj, sizeTemp));
	if (!Response_obj.data)
	{
		if (!IGNORE) cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}
	if (DEBUG_MODE) imshow("Response obj", Response_obj);



	Response_scene = _InputImage2(Rect(upLeftCorner_scene, sizeTemp));
	if (!Response_obj.data)
	{
		if (!IGNORE) cout << "can't create the image" << endl;
		return STATUS_ERROR;
	}

	if (DEBUG_MODE) imshow("Response scene", Response_scene);


	_BoxSize.clear();
	if (DEBUG_MODE)
	{
		if (!img_goodmatches.data != 1)
		{
			Mat img_matches_2 = img_goodmatches.clone();

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

			//-- Show detected matches
			imshow("Homography", img_matches_2);
			waitKey(30);
		}
		else if (!img_matches.data != 1)
		{
			Mat img_matches_2 = img_matches.clone();

			//-- Draw lines between the corners (the mapped object in the scene - image_2 )
			line(img_matches_2, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
			line(img_matches_2, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);

			//-- Show detected matches
			imshow("Homography", img_matches_2);
			waitKey(30);

		}

	}
	_BoxSize.push_back(sizeTemp.width);
	_BoxSize.push_back(sizeTemp.height);

	Response_obj.copyTo(_OutImage1);
	Response_scene.copyTo(_OutImage2);

	return STATUS_OK;
}

int SVM_HOG_Training::NegSampleWindowRetrieval_SIFT_SURF_SSIM(int _WindowsNumber,
	char* _SaveDir)
{
	Mat srcTemp;
	string ImgNameTemp;
	char NegSampleWindowSaveName[1000];				//Temp for name of negative sample window 
	char NegSampleWindowSaveNameForList[1000];
	int lineCount = 0;
	int SaveCount = 0;
	int ScanCount = 0;								//Count how many times of scan per image
	int threshold_keypoint = 25;					//For the first NegWin threshold
	double threshold_Box_Area = 0.3;

	double threshold_SSIM_Homo = 0.5;
	double threshold_SSIM_NonHomo = 0.7;

	ifstream finNeg(NegListDir);					//Load negative sample list
	int count = 0;

	char NegSampleWindowList[1000];
	sprintf_s(NegSampleWindowList, sizeof(NegSampleWindowList), "%sINRIANegWindowsList.txt", _SaveDir);
	ofstream foutNegWinList(NegSampleWindowList);

	while (getline(finNeg, ImgNameTemp))
	{

		if (PRINT_DETAILLY) std::cout << "Cut NegWindow, NegImg " << lineCount + 1 << "：" << ImgNameTemp << endl;
		ImgNameTemp = NegSampleDir + ImgNameTemp;
		srcTemp = imread(ImgNameTemp);
		if (!srcTemp.data)                      // Check for invalid input
		{
			cout << "Could not open or find the image in NegImg " << lineCount + 1 << endl;
			cout << "Please check the diectory is incorrect." << endl;
			cout << ImgNameTemp << endl;
			return STATUS_ERROR;
		}

		//图片大小应该能能至少包含一个64*128的窗口  
		if (srcTemp.cols >= 64 && srcTemp.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子
			count = 0;
			vector<cv::Mat> NegWin;
			while (true)
			{
				ScanCount++;
				int x = (rand() % (srcTemp.cols - 64));			//左上角x坐标  
				int y = (rand() % (srcTemp.rows - 128));			//左上角y坐标 
				Mat NegWinTemp = srcTemp(Rect(x, y, 64, 128));
				vector<KeyPoint> KeyPointTemp = KeyPointTest_SIFT(NegWinTemp);
				if (DEBUG_MODE) cout << KeyPointTemp.size() << " " << count << endl;

				if (!count)
				{
					if (KeyPointTemp.size() > threshold_keypoint)
					{
						count = 1;
						sprintf_s(NegSampleWindowSaveName, sizeof(NegSampleWindowSaveName), "%snoperson%06d.jpg",
							_SaveDir, SaveCount);//生成裁剪出的负样本图片的文件名 
						sprintf_s(NegSampleWindowSaveNameForList, sizeof(NegSampleWindowSaveNameForList), "noperson%06d.jpg",
							SaveCount);
						if (PRINT_DETAILLY) cout << NegSampleWindowSaveName << endl;
						imwrite(string(NegSampleWindowSaveName), NegWinTemp);//保存文件  
						foutNegWinList << NegSampleWindowSaveNameForList << endl;
						NegWin.push_back(NegWinTemp);

						SaveCount++;
						cout << "--------------Save Successfully----------------" << count << endl;
						if (DEBUG_MODE){}	//waitKey(0);
						threshold_keypoint = 25;
					}
				}
				else
				{
					int CompareCount = 0;

					/*
					***Detect Algorithm
					***Detect whether the new one is different with old the others
					*/
					for (size_t i = 0; i < NegWin.size(); i++)
					{

						Mat OutImage1, OutImage2;
						std::vector<int> Boxsize;
						int HomoStatus = HomographyTest_SIFT_SURF(NegWinTemp, NegWin[i], OutImage1, OutImage2, Boxsize);
						if (HomoStatus)											//have homography of the two image
						{
							cv::Scalar SSIM = calMSSIM(OutImage1, OutImage2);

							if (PRINT_DETAILLY) cout << "SSIM: " << SSIM << endl;

							if ((SSIM[0] < threshold_SSIM_Homo) && (SSIM[1] < threshold_SSIM_Homo) && (SSIM[2] < threshold_SSIM_Homo))
							{
								CompareCount++;
							}
							else
							{
								/*if the mapped region is out of 30 percent of whole, regard as similarity*/
								double TargetArea = Boxsize[0] * Boxsize[1];
								double WholeArea = 128 * 64;
								double ratio = TargetArea / WholeArea;
								if (ratio < threshold_Box_Area)
								{
									CompareCount++;
								}
							}
						}
						else
						{
							cv::Scalar SSIM = calMSSIM(NegWinTemp, NegWin[i]);

							if (PRINT_DETAILLY) cout << "SSIM: " << SSIM << endl;

							if ((SSIM[0] < threshold_SSIM_NonHomo) && (SSIM[1] < threshold_SSIM_NonHomo) && (SSIM[2] < threshold_SSIM_NonHomo))
							{
								CompareCount++;
							}
						}
						if (PRINT_DETAILLY) cout << "NegImg " << lineCount + 1 << " :" << " Scan: " << ScanCount << " Compare: " << CompareCount << "/" << i + 1 << endl;
						/*if it is similar to any one of the NegWin, then end the detect algorithm*/
						//cout << "NegImg " << lineCount + 1 << " :" << " Scan: " << ScanCount << " Compare: " << CompareCount << "/" << i + 1 << endl;
						if (CompareCount != (i + 1)) break;

					}

					if (PRINT_DETAILLY) cout << "CompareCount: " << CompareCount << " NegWin size: " << NegWin.size() << endl;

					/*Save the NegWinTemp as NegWin if it's different from the other image*/
					if (CompareCount == NegWin.size())
					{
						sprintf_s(NegSampleWindowSaveName, sizeof(NegSampleWindowSaveName), "%snoperson%06d.jpg",
							_SaveDir, SaveCount);								//生成裁剪出的负样本图片的文件名 
						sprintf_s(NegSampleWindowSaveNameForList, sizeof(NegSampleWindowSaveNameForList), "noperson%06d.jpg",
							SaveCount);
						if (PRINT_DETAILLY) cout << NegSampleWindowSaveName << endl;
						imwrite(string(NegSampleWindowSaveName), NegWinTemp);	//保存文件  
						foutNegWinList << NegSampleWindowSaveNameForList << endl;
						NegWin.push_back(NegWinTemp);

						SaveCount++;
						cout << "--------------Save Successfully----------------" << endl;
						if (DEBUG_MODE){}	//waitKey(0);
					}
				}

				if (ScanCount == 500)
				{
					srand(time(NULL));
					threshold_keypoint = 9;							//for some extreme scene
					//waitKey(0);
				}

				if (ScanCount == 1000)
				{
					threshold_SSIM_Homo = 0.55;
					threshold_SSIM_NonHomo = 0.75;
					//waitKey(0);
				}

				if (ScanCount == 2500)
				{
					threshold_SSIM_Homo = 0.6;
					threshold_SSIM_NonHomo = 0.8;
					//waitKey(0);
				}

				if (ScanCount == 3500)
				{
					threshold_SSIM_Homo = 0.65;
					threshold_SSIM_NonHomo = 0.85;
					//waitKey(0);
				}
				if (ScanCount == 4500)
				{
					srand(time(NULL));
					threshold_SSIM_Homo = 0.7;
					threshold_Box_Area = 0.33;
					//waitKey(0);
				}
				if (ScanCount == 5500)
				{
					threshold_Box_Area = 0.36;
					//waitKey(0);
				}
				if (ScanCount == 6500)
				{
					threshold_Box_Area = 0.36;
					//waitKey(0);
				}
				if (ScanCount == 7500)
				{
					srand(time(NULL));
					threshold_Box_Area = 0.39;
					//waitKey(0);
				}
				if (ScanCount == 8500)
				{
					threshold_Box_Area = 0.42;
					//waitKey(0);
				}
				if (ScanCount == 9500)
				{
					threshold_Box_Area = 0.48;
					//waitKey(0);
				}
				if (ScanCount == 10000)
				{
					srand(time(NULL));
					threshold_Box_Area = 0.5;
					//waitKey(0);
				}

				if (NegWin.size() == _WindowsNumber)			//Out when we have saved _WindowsNumber number of NegWin, and reset the threshold and param
				{
					cout << ScanCount << endl;
					ScanCount = 0;
					threshold_Box_Area = 0.3;
					threshold_SSIM_Homo = 0.5;
					threshold_SSIM_NonHomo = 0.7;
					//waitKey(0);
					break;
				}
				//system("pause");
			}

		}
		lineCount++;
	}

	return STATUS_OK;
}

int SVM_HOG_Training::NegSampleWindowRetrieval_SIFT_SURF_SSIM_M(int _WindowsNumber,
	char* _SaveDir, vector<string> _ImgList)
{
	Mat srcTemp;
	string ImgNameTemp;
	char NegSampleWindowSaveName[1000];				//Temp for name of negative sample window 
	char NegSampleWindowSaveNameForList[1000];
	int lineCount = 0;
	int SaveCount = 0;
	int ScanCount = 0;								//Count how many times of scan per image
	int InputCount = 0;								//Determinate when exit while loop

	int threshold_keypoint = 25;					//For the first NegWin threshold
	double threshold_Box_Area = 0.3;
	double threshold_SSIM_Homo = 0.5;
	double threshold_SSIM_NonHomo = 0.7;

	ifstream finNeg(NegListDir);					//Load negative sample list
	int count = 0;

	char NegSampleWindowList[1000];
	sprintf_s(NegSampleWindowList, sizeof(NegSampleWindowList), "%sINRIANegWindowsList.txt", _SaveDir);
	ofstream foutNegWinList(NegSampleWindowList);

	while (InputCount < _ImgList.size())
	{

		if (PRINT_DETAILLY) std::cout << "Cut NegWindow, NegImg " << lineCount + 1 << "：" << ImgNameTemp << endl;
		ImgNameTemp = NegSampleDir + ImgNameTemp;
		srcTemp = imread(ImgNameTemp);
		if (!srcTemp.data)                      // Check for invalid input
		{
			cout << "Could not open or find the image in NegImg " << lineCount + 1 << endl;
			cout << "Please check the diectory is incorrect." << endl;
			cout << ImgNameTemp << endl;
			return STATUS_ERROR;
		}

		//图片大小应该能能至少包含一个64*128的窗口  
		if (srcTemp.cols >= 64 && srcTemp.rows >= 128)
		{
			srand(time(NULL));//设置随机数种子
			count = 0;
			vector<cv::Mat> NegWin;
			while (true)
			{
				ScanCount++;
				int x = (rand() % (srcTemp.cols - 64));			//左上角x坐标  
				int y = (rand() % (srcTemp.rows - 128));			//左上角y坐标 
				Mat NegWinTemp = srcTemp(Rect(x, y, 64, 128));
				vector<KeyPoint> KeyPointTemp = KeyPointTest_SIFT(NegWinTemp);
				if (DEBUG_MODE) cout << KeyPointTemp.size() << " " << count << endl;

				if (!count)
				{
					if (KeyPointTemp.size() > threshold_keypoint)
					{
						count = 1;
						sprintf_s(NegSampleWindowSaveName, sizeof(NegSampleWindowSaveName), "%snoperson%06d.jpg",
							_SaveDir, SaveCount);//生成裁剪出的负样本图片的文件名 
						sprintf_s(NegSampleWindowSaveNameForList, sizeof(NegSampleWindowSaveNameForList), "noperson%06d.jpg",
							SaveCount);
						if (PRINT_DETAILLY) cout << NegSampleWindowSaveName << endl;
						imwrite(string(NegSampleWindowSaveName), NegWinTemp);//保存文件  
						foutNegWinList << NegSampleWindowSaveNameForList << endl;
						NegWin.push_back(NegWinTemp);

						SaveCount++;
						cout << "--------------Save Successfully----------------" << count << endl;
						if (DEBUG_MODE){}	//waitKey(0);
						threshold_keypoint = 25;
					}
				}
				else
				{
					int CompareCount = 0;

					/*
					***Detect Algorithm
					***Detect whether the new one is different with old the others
					*/
					for (size_t i = 0; i < NegWin.size(); i++)
					{

						Mat OutImage1, OutImage2;
						std::vector<int> Boxsize;
						int HomoStatus = HomographyTest_SIFT_SURF(NegWinTemp, NegWin[i], OutImage1, OutImage2, Boxsize);
						if (HomoStatus)											//have homography of the two image
						{
							cv::Scalar SSIM = calMSSIM(OutImage1, OutImage2);

							if (PRINT_DETAILLY) cout << "SSIM: " << SSIM << endl;

							if ((SSIM[0] < threshold_SSIM_Homo) && (SSIM[1] < threshold_SSIM_Homo) && (SSIM[2] < threshold_SSIM_Homo))
							{
								CompareCount++;
							}
							else
							{
								/*if the mapped region is out of 30 percent of whole, regard as similarity*/
								double TargetArea = Boxsize[0] * Boxsize[1];
								double WholeArea = 128 * 64;
								double ratio = TargetArea / WholeArea;
								if (ratio < threshold_Box_Area)
								{
									CompareCount++;
								}
							}
						}
						else
						{
							cv::Scalar SSIM = calMSSIM(NegWinTemp, NegWin[i]);

							if (PRINT_DETAILLY) cout << "SSIM: " << SSIM << endl;

							if ((SSIM[0] < threshold_SSIM_NonHomo) && (SSIM[1] < threshold_SSIM_NonHomo) && (SSIM[2] < threshold_SSIM_NonHomo))
							{
								CompareCount++;
							}
						}
						if (PRINT_DETAILLY) cout << "NegImg " << lineCount + 1 << " :" << " Scan: " << ScanCount << " Compare: " << CompareCount << "/" << i + 1 << endl;
						/*if it is similar to any one of the NegWin, then end the detect algorithm*/
						//cout << "NegImg " << lineCount + 1 << " :" << " Scan: " << ScanCount << " Compare: " << CompareCount << "/" << i + 1 << endl;
						if (CompareCount != (i + 1)) break;

					}

					if (PRINT_DETAILLY) cout << "CompareCount: " << CompareCount << " NegWin size: " << NegWin.size() << endl;

					/*Save the NegWinTemp as NegWin if it's different from the other image*/
					if (CompareCount == NegWin.size())
					{
						sprintf_s(NegSampleWindowSaveName, sizeof(NegSampleWindowSaveName), "%snoperson%06d.jpg",
							_SaveDir, SaveCount);								//生成裁剪出的负样本图片的文件名 
						sprintf_s(NegSampleWindowSaveNameForList, sizeof(NegSampleWindowSaveNameForList), "noperson%06d.jpg",
							SaveCount);
						if (PRINT_DETAILLY) cout << NegSampleWindowSaveName << endl;
						imwrite(string(NegSampleWindowSaveName), NegWinTemp);	//保存文件  
						foutNegWinList << NegSampleWindowSaveNameForList << endl;
						NegWin.push_back(NegWinTemp);

						SaveCount++;
						cout << "--------------Save Successfully----------------" << endl;
						if (DEBUG_MODE){}	//waitKey(0);
					}
				}

				if (ScanCount == 500)
				{
					srand(time(NULL));
					threshold_keypoint = 9;							//for some extreme scene
					//waitKey(0);
				}

				if (ScanCount == 1000)
				{
					threshold_SSIM_Homo = 0.55;
					threshold_SSIM_NonHomo = 0.75;
					//waitKey(0);
				}

				if (ScanCount == 2500)
				{
					threshold_SSIM_Homo = 0.6;
					threshold_SSIM_NonHomo = 0.8;
					//waitKey(0);
				}

				if (ScanCount == 3500)
				{
					threshold_SSIM_Homo = 0.65;
					threshold_SSIM_NonHomo = 0.85;
					//waitKey(0);
				}
				if (ScanCount == 4500)
				{
					srand(time(NULL));
					threshold_SSIM_Homo = 0.7;
					threshold_Box_Area = 0.33;
					//waitKey(0);
				}
				if (ScanCount == 5500)
				{
					threshold_Box_Area = 0.36;
					//waitKey(0);
				}
				if (ScanCount == 6500)
				{
					threshold_Box_Area = 0.36;
					//waitKey(0);
				}
				if (ScanCount == 7500)
				{
					srand(time(NULL));
					threshold_Box_Area = 0.39;
					//waitKey(0);
				}
				if (ScanCount == 8500)
				{
					threshold_Box_Area = 0.42;
					//waitKey(0);
				}
				if (ScanCount == 9500)
				{
					threshold_Box_Area = 0.48;
					//waitKey(0);
				}
				if (ScanCount == 10000)
				{
					srand(time(NULL));
					threshold_Box_Area = 0.5;
					//waitKey(0);
				}

				if (NegWin.size() == _WindowsNumber)			//Out when we have saved _WindowsNumber number of NegWin, and reset the threshold and param
				{
					cout << ScanCount << endl;
					ScanCount = 0;
					threshold_Box_Area = 0.3;
					threshold_SSIM_Homo = 0.5;
					threshold_SSIM_NonHomo = 0.7;
					//waitKey(0);
					break;
				}
				//system("pause");
			}

		}
		lineCount++;
	}

	return STATUS_OK;
}

int SVM_HOG_Training::NegSampleWindowRetrievalHomoTest(char* _NegWindowsDir, char* _NegWindowsListDir)
{
	ifstream finNegWin(_NegWindowsListDir);

	string ImgNameTemp;
	vector<string> imgList;

	while (getline(finNegWin, ImgNameTemp))
	{
		ImgNameTemp = _NegWindowsDir + ImgNameTemp;
		imgList.push_back(ImgNameTemp);
	}
	finNegWin.close();

	Mat out1, out2;
	std::vector<int> Boxsize;
	int NegNumber = imgList.size() / 10;
	cout << NegNumber << endl;
	for (size_t k = 0; k < NegNumber; k++)
	{
		for (size_t i = 1 + (k * 10); i < 10 + (k * 10); i++)
		{
			for (size_t j = k * 10; j < i; j++)
			{
				cout << "Compare: " << i << "to" << j << endl;
				int result = HomographyTest_SIFT_SURF(imread(imgList[i]), imread(imgList[j]), out1, out2, Boxsize);
				if (result)
				{
					cout << "Have Homo" << endl;
					cv::Scalar SSIM = calMSSIM(out1, out2);
					cout << "SSIM: "<< SSIM << endl;

					double TargetArea = Boxsize[0] * Boxsize[1];
					double WholeArea = 128 * 64;
					double ratio = TargetArea / WholeArea;
					cout << "mapped area ratio: " << ratio << endl;
				}
				else
				{
					cout << "Have no Homo" << endl;
					cv::Scalar SSIM = calMSSIM(imread(imgList[i]), imread(imgList[j]));
					cout << SSIM << endl;
				}
				waitKey(0);
			}

		}
	}

}

void SVM_HOG_Training::SVM_Train_Demo()
{
	// Data for visual representation
	int width = 512, height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	int labels[4] = { 1, -1, -1, -1 };
	Mat labelsMat(4, 1, CV_32SC1, labels);

	float trainingData[4][2] = { { 501, 10 }, { 255, 10 }, { 501, 255 }, { 10, 501 } };
	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	Ptr<SVM> mySVM = cv::ml::SVM::create();
	// Set up SVM's parameters
	mySVM->setType(SVM::C_SVC);
	mySVM->setKernel(SVM::LINEAR);
	mySVM->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, 100, 1e-6));

	// Train the SVM
	mySVM->train(trainingDataMat, ml::ROW_SAMPLE, labelsMat);

	Vec3b green(0, 255, 0), blue(255, 0, 0);
	// Show the decision regions given by the SVM
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
			float response = mySVM->predict(sampleMat);

			if (response == 1)
				image.at<Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<Vec3b>(i, j) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	Mat sv = mySVM->getSupportVectors();
	cout << sv.rows << "" << sv.cols << endl;
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(0, 0, 0), thickness, lineType);
	}

	imwrite("result.png", image);        // save the image

	imshow("SVM Simple Example", image); // show it to the user
	waitKey(0);
}

void SVM_HOG_Training::SIFT_DAISY_Demo()
{
	const float nn_match_ratio = 0.7f;
	const float keypoint_diameter = 15.0f;

	// Load images
	Mat img1 = imread("test001.jpg");
	Mat img2 = imread("test002.jpg");

	vector<KeyPoint> keypoints1, keypoints2;

	// Add every pixel to the list of keypoints for each image

	cv::Ptr<xfeatures2d::SIFT> SIFT = xfeatures2d::SIFT::create();

	SIFT->detect(img1, keypoints1);
	SIFT->detect(img2, keypoints2);

	Mat desc1, desc2;

	Ptr<cv::xfeatures2d::DAISY> descriptor_extractor = cv::xfeatures2d::DAISY::create();

	// Compute DAISY descriptors for both images 
	descriptor_extractor->compute(img1, keypoints1, desc1);
	descriptor_extractor->compute(img2, keypoints2, desc2);

	vector <vector<DMatch>> matches;

	// For each descriptor in image1, find 2 closest matched in image2 (note: couldn't get BF matcher to work here at all)
	FlannBasedMatcher flannmatcher;
	flannmatcher.add(desc1);
	flannmatcher.train();
	flannmatcher.knnMatch(desc2, matches, 2);


	// ignore matches with high ambiguity -- i.e. second closest match not much worse than first
	// push all remaining matches back into DMatch Vector "good_matches" so we can draw them using DrawMatches
	int                 num_good = 0;
	vector<KeyPoint>    matched1, matched2;
	vector<DMatch>      good_matches;

	for (int i = 0; i < matches.size(); i++) {
		DMatch first = matches[i][0];
		DMatch second = matches[i][1];

		if (first.distance < nn_match_ratio * second.distance) {
			matched1.push_back(keypoints1[first.trainIdx]);
			matched2.push_back(keypoints2[first.queryIdx]);
			good_matches.push_back(DMatch(num_good, num_good, 0));
			num_good++;
		}
	}

	Mat res;
	drawMatches(img1, matched1, img2, matched2, good_matches, res);
	imshow("123", res);
	waitKey(30);
}

void SVM_HOG_Training::HomographyTest_SIFT_SURF_Demo()
{
	ifstream finXX("PSNRTest/TestList.txt");
	vector<string> detector;
	char imgName[500];

	std::vector<int> Boxsize;
	string val;
	while (!finXX.eof())
	{
		finXX >> val;
		cout << val << endl;
		sprintf_s(imgName, sizeof(imgName), "PSNRTest/%s", val.c_str());
		cout << imgName << endl;
		detector.push_back(imgName);
	}
	finXX.close();

	for (size_t i = 0; i < detector.size() / 2; i++)
	{
		cout << detector[2 * i] << endl;
		imshow("1", imread(detector[2 * i]));
		cout << detector[2 * i + 1] << endl;
		imshow("2", imread(detector[2 * i + 1]));

		cout << KeyPointTest_SIFT(imread(detector[2 * i])).size() << endl;
		cout << KeyPointTest_SIFT(imread(detector[2 * i + 1])).size() << endl;
		Mat out1, out2;
		int result = HomographyTest_SIFT_SURF(imread(detector[2 * i]), imread(detector[2 * i + 1]), out1, out2, Boxsize);
		if (result)
		{
			cv::Scalar SSIM = calMSSIM(out1, out2);
			cout << SSIM << endl;
		}
		else
		{
			cv::Scalar SSIM = calMSSIM(imread(detector[2 * i]), imread(detector[2 * i + 1]));
			cout << SSIM << endl;
		}

		waitKey(0);
		destroyAllWindows();
	}
}