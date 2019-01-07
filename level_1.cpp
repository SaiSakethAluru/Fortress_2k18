#include <iostream>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <bits/stdc++.h>
#include <cmath>
#include <time.h>

using namespace std;
using namespace cv;

int pixel_count(Mat &img);
pair<int,int> make_pair(int i, int j);
Mat binary(Mat &img);
Mat create_template(Mat &img);

typedef struct _color{
	int B,G,R;
	int b,g,r;
}color;


int main()
{
	// clocks for recording execution time. 
	clock_t c1,c2;
	double runtime;
	c1= clock();   // start stopwatch. 
	// Threshold values of all colours. 
	// TO BE ADJUSTED DURING MATCH
	color blue,green,red,yellow;
	blue.B=255;blue.G=100;blue.R=100;
	blue.b=200;blue.g=0;blue.r=0;
	green.B=100;green.G=255;green.R=100;
	green.b=0;green.g=150;green.r=0;
	red.B=100;red.G=100;red.R=255;
	red.b=0;red.g=0;red.r=200;
	yellow.B=100;yellow.G=255;yellow.R=255;
	yellow.b=0;yellow.g=150;yellow.r=150;
	//Priority order of colours.
	//ADJUST DEPENDING ON QUESTION. 
	color order[4] = {blue,green,red,yellow};
	
	// Actually should be VideoCapture(0) and vid.read()
	Mat videoImg = imread("image_1.png");
	// converting image to binary scale.
	// ADJUST THRESHOLD IN FUNCTION DURING MATCH.
	Mat arenaImg = binary(videoImg); 
	imshow("arena",arenaImg);
	imwrite("newbin.png",arenaImg);
	waitKey(0);
	/*counting pixels in one blob of arena image. 
	Approximately same for all. */
	int apix = pixel_count(arenaImg);
	cout<<"no. of pixels = "<<apix<<endl;
	// Read the given colour image.
	Mat colorImg = imread("image_2.png");
	// Vector containing the locations of blocks in priority order. 
	vector<pair <int, int> > locs;
	for(int k=0;k<4;k++){
		//stores individual colour block in binary form.
		Mat singleImg(colorImg.rows,colorImg.cols,CV_8UC1,Scalar(0));
		for(int i=0;i<singleImg.rows;i++){
			for(int j=0;j<singleImg.cols;j++){
				if(colorImg.at<Vec3b>(i,j)[0] >= order[k].b && colorImg.at<Vec3b>(i,j)[0] <= order[k].B && colorImg.at<Vec3b>(i,j)[1] >= order[k].g && colorImg.at<Vec3b>(i,j)[1] <= order[k].G && colorImg.at<Vec3b>(i,j)[2] <= order[k].R && colorImg.at<Vec3b>(i,j)[2] >= order[k].r){
					singleImg.at<uchar>(i,j) = 255;		
				}
			}
		}
		int cpix = 0;					// No. of pixels in each individual blob.
		cpix = pixel_count(singleImg);
		cout<<"For blob "<<k+1<<" no. of pixels = "<<cpix<<endl;
		// Taking square root for ratio as we are adjusting both length and breadth.
		double ratio = sqrt((double)apix/(double)cpix);
		// Resize the blocks to size of arena block.
		resize(singleImg,singleImg,Size(),ratio,ratio);
		// Extracting block as a template. 
		Mat block = create_template(singleImg);
		// result Mat is temporary; used during template matching.
		int rows = arenaImg.rows-block.rows+1;
		int cols = arenaImg.cols-block.cols+1;
		Mat result(rows,cols,CV_8UC1);
		double minVal,maxVal;
		matchTemplate(arenaImg,block,result,CV_TM_SQDIFF);
		normalize(result,result,0,1,NORM_MINMAX,-1,Mat());
		Point minLoc,maxLoc;
		minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc,Mat());
		// After above step, minLoc contains the location of match. 
		int xcord = minLoc.x+block.rows/2;
		int ycord = minLoc.y+block.cols/2;
		cout<<"Location of block "<<k+1<<" is "<<xcord<<" "<<ycord<<endl;
		locs.push_back(make_pair(xcord,ycord));
	}
	c2 = clock();   // end stopwatch.
	runtime = (double)(c2-c1)/(double)CLOCKS_PER_SEC;
	cout<<"running time: "<<runtime<<endl;
	return 0;
}
/*
Doing blob detection to count no. of pixels in the image.
Blob detection is opted as we don't need to traverse all points in image this way.

*/
int pixel_count(Mat &img)
{
	stack <pair <int, int > > s;
	int count=0;
	// int v[img.rows][img.cols];
	Mat visit(img.rows,img.cols,CV_8UC1,Scalar(0));
	// for(int i=0;i<img.rows;i++){
	// 	for(int j=0;j<img.cols;j++){
	// 		v[i][j]=0;
	// 	}
	// }
	for(int i=0;i<img.rows;i++){
		for(int j=0;j<img.cols;j++){
			if(img.at<uchar>(i,j)>100 && visit.at<uchar>(i,j)==0){
				s.push(make_pair(i,j));
				count++;
				// v[i][j]=1;
				visit.at<uchar>(i,j) = 1;
				while(!s.empty()){
					pair<int, int> p = s.top();
					s.pop();
					for(int k=p.first-1;k<p.first+2;k++){
						for(int l=p.second-1;l<p.second+2;l++){
							if(k>=0 && k<img.rows && l>=0 && l<img.cols){	
								if(visit.at<uchar>(k,l) == 0 && img.at<uchar>(k,l)>100){
									// v[k][l]=1;
									visit.at<uchar>(k,l) = 1;
									s.push(make_pair(k,l));
									count++;
								}
							}
						}
					}
				}
				return count;
			}
		}
	}
	return -1;     // exception handling. Ignore for now. 
}

/*
Function to make pair out of two integers
Used in blob detection part of pixel_count function.
*/
pair<int,int> make_pair(int i, int j)
{
    pair<int, int> p;
    p.first = i;
    p.second = j;
    return p;
}
/*
Function to convert image to binary. 
THRESHOLD TO BE ADJUSTED DURING PLAY.
Conversion is done kernel wise to simultaneously include erosion. 
This is because given image has some spaces in between the blocks. 
*/
Mat binary(Mat &img)
{
	Mat result(img.rows,img.cols,CV_8UC1,Scalar(0));
	int kernel = 5;   // kernel size to change at a time. 5 works for given image. To be adjusted based on requirement. 
	color threshold;
	threshold.b=0;threshold.B=100;
	threshold.g=150;threshold.G=255;
	threshold.r=150;threshold.R=255;

	for(int i=0;i<img.rows-kernel;i+=kernel){
		for(int j=0;j<img.cols-kernel;j+=kernel){
			vector<int> v;
			int flag = 0;
			for(int k=i;k<i+kernel;k++){
				for(int l=j;l<j+kernel;l++){
					if(!flag && img.at<Vec3b>(k,l)[0]>=threshold.b && img.at<Vec3b>(k,l)[0]<=threshold.B && img.at<Vec3b>(k,l)[1]>=threshold.g && img.at<Vec3b>(k,l)[1]<=threshold.G && img.at<Vec3b>(k,l)[2]>=threshold.r && img.at<Vec3b>(k,l)[2]<=threshold.R){
						v.push_back(255);
						flag = 1;
					}
					else v.push_back(0);
				}
			}
			int a = *max_element(v.begin(), v.end());
			for(int k=i;k<i+kernel;k++){
				for(int l=j;l<j+kernel;l++){
					result.at<uchar>(k,l) = a;
				}
			}
		}
	}
	return result;
}
/*
Function to extract out the template out of individual blocks from the given image. 
This template is used in matchTemplate.
*/
Mat create_template(Mat &img)
{
	int r = img.rows;
	int c = img.cols;
	int r1,r2,c1,c2;
	int flag=0;
	int flagprev = 0;
	for(int j=0;j<c;j++){
		flag = 0;
		for(int i=0;i<r;i++){
			if(img.at<uchar>(i,j) > 128){
				flag = 1;
				break;
			}
		}
		if(flag>flagprev){
			c1 = j;
		}
		else if(flag<flagprev){
			c2 = j;
			break;
		}
		flagprev = flag;
	}
	flag = flagprev = 0;
	for(int i=0;i<r;i++){
		flag = 0;
		for(int j=c1;j<=c2;j++){
			if(img.at<uchar>(i,j)>128){
				flag = 1;
			}
		}
		if(flag>flagprev)
		{
			r1 = i;
		}
		else if(flag<flagprev){
			r2 = i;
			break;
		}
		flagprev = flag;
	}
	Mat img2(r2-r1+1,c2-c1+1,CV_8UC1,Scalar(0));
	for(int i=r1;i<=r2;i++){
		for(int j=c1;j<=c2;j++){
			img2.at<uchar>(i-r1,j-c1) = img.at<uchar>(i,j);
		}
	}
	return img2;	
}