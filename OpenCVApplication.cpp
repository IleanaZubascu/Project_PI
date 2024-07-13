// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <queue>
#include <random>

default_random_engine gen;
uniform_int_distribution<int> d(0, 255);


wchar_t* projectPath;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	_wchdir(projectPath);

	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	waitKey(0);
}



boolean isInside(Mat img, int i, int j)
{
	
	return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
 
}



void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Accessing individual pixels in an 8 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testNegativeImageFast()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// The fastest approach of accessing the pixels -> using pointers
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Accessing individual pixels in a RGB 24 bits/pixel image
		// Inefficient way -> slow
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}


void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	_wchdir(projectPath);

	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(100);  // waits 100ms and advances to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	_wchdir(projectPath);

	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snap the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}



//PROIECT/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Mat etichetare(Mat src)
{

		int label = 0;
		int height, width;

		height = src.rows;
		width = src.cols;

		Mat labels = Mat::zeros(height, width, CV_8UC1);


		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
				{
					label++;
					queue<pair<int, int>> Q;

					labels.at<uchar>(i, j) = label;
					Q.push(pair<int, int>(i, j));



					while (!Q.empty())
					{
						pair<int, int> q = Q.front();
						Q.pop();
						int qi = q.first, gj = q.second;


						for (int dx = -1; dx <= 1; dx++)
						{
							for (int dy = -1; dy <= 1; dy++)
							{
								int ni = qi + dx;
								int nj = gj + dy;

								if (isInside(src, ni, nj) && src.at<uchar>(ni, nj) == 0 && labels.at<uchar>(ni, nj) == 0)
								{
									labels.at<uchar>(ni, nj) = label;
									Q.push(pair<int, int>(ni, nj));
								}
							}
						}
					}
				}
			}
		}

		
		return labels;
}


int NrOfObjects(Mat src)
{

	int label = 0;
	int height, width;

	height = src.rows;
	width = src.cols;

	Mat labels = Mat::zeros(height, width, CV_8UC1);


	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0)
			{
				label++;
				queue<pair<int, int>> Q;

				labels.at<uchar>(i, j) = label;
				Q.push(pair<int, int>(i, j));



				while (!Q.empty())
				{
					pair<int, int> q = Q.front();
					Q.pop();
					int qi = q.first, gj = q.second;


					for (int dx = -1; dx <= 1; dx++)
					{
						for (int dy = -1; dy <= 1; dy++)
						{
							int ni = qi + dx;
							int nj = gj + dy;

							if (isInside(src, ni, nj) && src.at<uchar>(ni, nj) == 0 && labels.at<uchar>(ni, nj) == 0)
							{
								labels.at<uchar>(ni, nj) = label;
								Q.push(pair<int, int>(ni, nj));
							}
						}
					}
				}
			}
		}
	}


	return label;
}





Mat ImagineCuContur(Mat src, Mat labels, int label)
{
	int height, width;
	height = src.rows;
	width = src.cols;

	Mat dst = Mat(height, width, CV_8UC3);
	Mat contur = Mat(height, width, CV_8UC1, Scalar(255));


	Vec3b color[1000];

	for (int i = 0; i < 1000; i++)
		color[i] = Vec3b(d(gen), d(gen), d(gen));


	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{

				dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
			}
			else dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

		}

	for (int k = 1; k <= label; k++)
	{
		int x = 0, y = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				if (labels.at<uchar>(i, j) == k)
				{
					x = i;
					y = j;
					i = height;
					break;
				}

			}

		int dir = 7, start_dir;
		int cod[10000] = { 0 };
		int cod_index = 0;
		cod[cod_index] = dir;


		int firstX = x, firstY = y, secondX, secondY;


		if (dir % 2 == 0)
			start_dir = (dir + 7) % 8;
		else start_dir = (dir + 6) % 8;

		Point v[9];
		v[0].x = x; v[0].y = y + 1;
		v[1].x = x - 1; v[1].y = y + 1;
		v[2].x = x - 1; v[2].y = y;
		v[3].x = x - 1; v[3].y = y - 1;
		v[4].x = x; v[4].y = y - 1;
		v[5].x = x + 1; v[5].y = y - 1;
		v[6].x = x + 1; v[6].y = y;
		v[7].x = x + 1; v[7].y = y + 1;

		if (src.at<uchar>(v[start_dir].x, v[start_dir].y) == 0)
		{

			cod[++cod_index] = start_dir;
			dir = start_dir;

			secondX = v[start_dir].x;
			secondY = v[start_dir].y;

			dst.at<Vec3b>(v[start_dir].x, v[start_dir].y) = color[labels.at<uchar>(x, y)];
			contur.at<uchar>(v[start_dir].x, v[start_dir].y) = 0;


		}
		else
		{
			while (src.at<uchar>(v[start_dir].x, v[start_dir].y) != 0)
			{
				start_dir = (start_dir + 1) % 8;
			}
			cod[++cod_index] = start_dir;
			dir = start_dir;
			secondX = v[start_dir].x;
			secondY = v[start_dir].y;

			dst.at<Vec3b>(v[start_dir].x, v[start_dir].y) = color[labels.at<uchar>(x, y)];
			contur.at<uchar>(v[start_dir].x, v[start_dir].y) = 0;
		}


		x = secondX;
		y = secondY;


		bool ok = false;
		int x_ant = x, y_ant = y;



		do
		{
			if (x == secondX && y == secondY && x_ant == firstX && y_ant == firstY) ok = true;

			if (dir % 2 == 0)
				start_dir = (dir + 7) % 8;
			else start_dir = (dir + 6) % 8;

			Point v[9];
			v[0].x = x; v[0].y = y + 1;
			v[1].x = x - 1; v[1].y = y + 1;
			v[2].x = x - 1; v[2].y = y;
			v[3].x = x - 1; v[3].y = y - 1;
			v[4].x = x; v[4].y = y - 1;
			v[5].x = x + 1; v[5].y = y - 1;
			v[6].x = x + 1; v[6].y = y;
			v[7].x = x + 1; v[7].y = y + 1;

			if (src.at<uchar>(v[start_dir].x, v[start_dir].y) == 0 )
			{

				cod[++cod_index] = start_dir;
				dir = start_dir;

				x_ant = x;
				y_ant = y;
				x = v[dir].x;
				y = v[dir].y;

				dst.at<Vec3b>(v[start_dir].x, v[start_dir].y) = color[labels.at<uchar>(x, y)];;
				contur.at<uchar>(v[start_dir].x, v[start_dir].y) = 0;


			}
			else
			{
				while (src.at<uchar>(v[start_dir].x, v[start_dir].y) != 0)
				{
					start_dir = (start_dir + 1) % 8;
				}
				cod[++cod_index] = start_dir;
				dir = start_dir;

				x_ant = x;
				y_ant = y;

				x = v[dir].x;
				y = v[dir].y;

				dst.at<Vec3b>(v[start_dir].x, v[start_dir].y) = color[labels.at<uchar>(x, y)];
				contur.at<uchar>(v[start_dir].x, v[start_dir].y) = 0;

			}



		} while (ok != true);

	}

	imshow("Labeled", dst);

	return contur;
	
}

Mat DT(Mat src)
{

	// wHV = 2 si wD = 3

	int height, width;
	height = src.rows;
	width = src.cols;

	Mat dt = Mat(height, width, CV_8UC1, Scalar(255));
	dt = src.clone();

	int mask[3][3] = { 0 };

	mask[0][0] = 3;
	mask[0][1] = 2;
	mask[0][2] = 3;
	mask[1][0] = 2;
	mask[1][2] = 2;
	mask[2][0] = 3;
	mask[2][1] = 2;
	mask[2][2] = 3;


	//prima parcurgere
	for(int i=1; i<height-1;i++)
		for (int j = 1; j < width-1; j++)
		{
			
				int minn = 260;


				if ((dt.at<uchar>(i - 1, j - 1) + mask[0][0]) < minn) minn = dt.at<uchar>(i - 1, j - 1) + mask[0][0];
				if ((dt.at<uchar>(i - 1, j) + mask[0][1]) < minn) minn = dt.at<uchar>(i - 1, j) + mask[0][1];
				if ((dt.at<uchar>(i - 1, j + 1) + mask[0][2]) < minn) minn = dt.at<uchar>(i - 1, j + 1) + mask[0][2];
				if ((dt.at<uchar>(i, j - 1) + mask[1][0]) < minn) minn = dt.at<uchar>(i, j - 1) + mask[1][0];

				dt.at<uchar>(i, j) = min(minn, dt.at<uchar>(i,j));
			
		}

	//a doua parcurgere

	for (int i = height - 2; i>0; i--)
		for (int j = width - 2; j > 0; j--)
		{
			


				int minn = 260;

				if ((dt.at<uchar>(i + 1, j + 1) + mask[2][2]) < minn) minn = dt.at<uchar>(i + 1, j + 1) + mask[2][2];
				if ((dt.at<uchar>(i + 1, j) + mask[2][1]) < minn) minn = dt.at<uchar>(i + 1, j) + mask[2][1];
				if ((dt.at<uchar>(i + 1, j - 1) + mask[2][0]) < minn) minn = dt.at<uchar>(i + 1, j - 1) + mask[2][0];
				if ((dt.at<uchar>(i, j + 1) + mask[1][2]) < minn) minn = dt.at<uchar>(i, j + 1) + mask[1][2];

				dt.at<uchar>(i, j) = min(minn, dt.at<uchar>(i,j));
			
		}
	return dt;


}

void conturImagineDoi(Mat src, Mat dt, Mat input1)
{

	int height, width;
		height = src.rows;
		width = src.cols;
		Mat dst = Mat(input1.rows, input1.cols, CV_8UC3);
		Mat input2 = Mat(height, width, CV_8UC3);

		for (int i = 0; i < input1.rows; i++)
			for (int j = 0; j < input1.cols; j++)
				if (input1.at<uchar>(i, j) == 0) dst.at<Vec3b>(i, j) = (0, 0, 0);
				else dst.at<Vec3b>(i, j) = Vec3b(255, 255, 255);


		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{

					input2.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
				}
				else input2.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

			}

		int x = 0, y = 0;
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				if (src.at<uchar>(i, j) == 0)
				{
					x = i;
					y = j;
					i = height;
					break;
				}

			}

		int dir = 7, start_dir;
		int* cod =(int*) malloc(1000*sizeof(int));
		int cod_index = 0;

		cod[cod_index] = dir;


		int firstX = x, firstY = y, secondX, secondY;


		if (dir % 2 == 0)
			start_dir = (dir + 7) % 8;
		else start_dir = (dir + 6) % 8;

		Point v[9];
		v[0].x = x; v[0].y = y + 1;
		v[1].x = x - 1; v[1].y = y + 1;
		v[2].x = x - 1; v[2].y = y;
		v[3].x = x - 1; v[3].y = y - 1;
		v[4].x = x; v[4].y = y - 1;
		v[5].x = x + 1; v[5].y = y - 1;
		v[6].x = x + 1; v[6].y = y;
		v[7].x = x + 1; v[7].y = y + 1;

		if (src.at<uchar>(v[start_dir].x, v[start_dir].y) == 0)
		{

			cod[++cod_index] = start_dir;
			dir = start_dir;

			secondX = v[start_dir].x;
			secondY = v[start_dir].y;

			input2.at<Vec3b>(v[start_dir].x, v[start_dir].y) = Vec3b(255, 255, 0);


		}
		else
		{
			while (src.at<uchar>(v[start_dir].x, v[start_dir].y) != 0)
			{
				start_dir = (start_dir + 1) % 8;
			}
			cod[++cod_index] = start_dir;
			dir = start_dir;
			secondX = v[start_dir].x;
			secondY = v[start_dir].y;

			input2.at<Vec3b>(v[start_dir].x, v[start_dir].y) = Vec3b(255, 255, 0);

		}


		x = secondX;
		y = secondY;


		bool ok = false;
		int x_ant = x, y_ant = y;

		do
		{
			if (x == secondX && y == secondY && x_ant == firstX && y_ant == firstY) ok = true;

			if (dir % 2 == 0)
				start_dir = (dir + 7) % 8;
			else start_dir = (dir + 6) % 8;

			Point v[9];
			v[0].x = x; v[0].y = y + 1;
			v[1].x = x - 1; v[1].y = y + 1;
			v[2].x = x - 1; v[2].y = y;
			v[3].x = x - 1; v[3].y = y - 1;
			v[4].x = x; v[4].y = y - 1;
			v[5].x = x + 1; v[5].y = y - 1;
			v[6].x = x + 1; v[6].y = y;
			v[7].x = x + 1; v[7].y = y + 1;

			if (src.at<uchar>(v[start_dir].x, v[start_dir].y) == 0)
			{

				cod[++cod_index] = start_dir;
				dir = start_dir;

				x_ant = x;
				y_ant = y;
				x = v[dir].x;
				y = v[dir].y;

				input2.at<Vec3b>(v[start_dir].x, v[start_dir].y) = Vec3b(255, 255, 0);


			}
			else
			{
				while (src.at<uchar>(v[start_dir].x, v[start_dir].y) != 0)
				{
					start_dir = (start_dir + 1) % 8;
				}
				cod[++cod_index] = start_dir;
				dir = start_dir;

				x_ant = x;
				y_ant = y;

				x = v[dir].x;
				y = v[dir].y;

				input2.at<Vec3b>(v[start_dir].x, v[start_dir].y) = Vec3b(255, 255, 0);

			}



		} while (ok != true);

		int heightDT = dt.rows, widthDT = dt.cols;
		int minn = dt.rows*dt.cols;
		Point result;

		for (int i = 0; i < heightDT; i++) {
			for (int j = 0; j < widthDT; j++) {
				int scor = 0;
				int iDT = i, jDT = j;
				int ok = 1;
				for (int k = 1; k < cod_index - 1; k++)
				{
					if (ok == 1)
					{
						switch (cod[k]) {
						case 0: {
							if (isInside(dt, iDT, jDT + 1))
							{
								scor += dt.at<uchar>(iDT, jDT + 1); jDT++;
							}
							else ok = 0;
							break;
						}
						case 1:
						{
							if (isInside(dt, iDT - 1, jDT + 1))
							{
								scor += dt.at<uchar>(iDT - 1, jDT + 1); iDT--; jDT++;
							}
							else ok = 0;
							break;
						}
						case 2:
						{
							if (isInside(dt, iDT - 1, jDT))
							{
								scor += dt.at<uchar>(iDT - 1, jDT); iDT--;
							}
							else ok = 0;
							break;
						}
						case 3: {
							if (isInside(dt, iDT - 1, jDT - 1))
							{
								scor += dt.at<uchar>(iDT - 1, jDT - 1); iDT--; jDT--;
							}
							else ok = 0;
							break;
						}
						case 4: {
							if (isInside(dt, iDT, jDT - 1))
							{
								scor += dt.at<uchar>(iDT, jDT - 1); jDT--;
							}
							else ok = 0;
							break;
						}
						case 5: {
							if (isInside(dt, iDT + 1, jDT - 1))
							{
								scor += dt.at<uchar>(iDT + 1, jDT - 1); iDT++; jDT--;
							}
							else ok = 0;
							break;
						}
						case 6: {
							if (isInside(dt, iDT + 1, jDT))
							{
								scor += dt.at<uchar>(iDT + 1, jDT); iDT++;
							}
							else ok = 0;
							break;
						}
						case 7: {
							if (isInside(dt, iDT + 1, jDT + 1))
							{
								scor += dt.at<uchar>(iDT + 1, jDT + 1); iDT++; jDT++;
							}
							else ok = 0;
							break;
						}
						}
					}
					else break;

				}
				if (minn > scor && ok == 1)
				{
					minn = scor;
					result.x = i;
					result.y = j;
				}
			}
		}

			for (int k = 1; k < cod_index - 1; k++)
			{
				switch (cod[k]) {
				case 0: {
					dst.at<Vec3b>(result.x, result.y + 1) = Vec3b(0, 0, 255);
					result.y++;
					break; }
				case 1:
				{dst.at<Vec3b>(result.x - 1, result.y + 1) = Vec3b(0, 0, 255);
				result.x--; result.y++;
				break; }
				case 2:
				{dst.at<Vec3b>(result.x - 1, result.y) = Vec3b(0, 0, 255);
				result.x--;
				break;
				}
				case 3:
				{dst.at<Vec3b>(result.x - 1, result.y - 1) = Vec3b(0, 0, 255);
				result.x--; result.y--;
				break;
				}
				case 4:
				{dst.at<Vec3b>(result.x, result.y - 1) = Vec3b(0, 0, 255);
				result.y--;
				break; }
				case 5:
				{dst.at<Vec3b>(result.x + 1, result.y - 1) = Vec3b(0, 0, 255);
				result.x++; result.y--;
				break; }
				case 6:
				{dst.at<Vec3b>(result.x + 1, result.y) = Vec3b(0, 0, 255);
				result.x++;
				break; }
				case 7:
				{dst.at<Vec3b>(result.x + 1, result.y + 1) = Vec3b(0, 0, 255);
				result.x++; result.y++;
				break; }
				}

			}
		




		cout <<"\nScore: "<< minn;

		
		imshow("Result", dst);

		imshow("Input2 with contour", input2);


}




void imagineaDoi(Mat DT, Mat dst)
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		 conturImagineDoi(src, DT, dst);




	}



}




void Proiect()
{

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);

		Mat labels = etichetare(src);

		int label = NrOfObjects(src);

		Mat dst = ImagineCuContur(src, labels, label);

		Mat dt = DT(dst);

		//imshow("src", src);
		imshow("DT", dt);

	    imagineaDoi(dt, src);





		waitKey();


		destroyAllWindows();
	}

}



int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("100 - Proiect\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 100:
				Proiect();
				break;

				
		}
	}
	while (op!=0);
	return 0;
}