///////////////////////////////////////////////////////////////////////////
/// reference to : http://www.opencv.org.cn
/// referencr to : hello-world.cpp
///
//////////////////////////////////////////////////////////////////////////

#define intest 1 //if in debug mode, set intest to 1, or set to 0
#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<cxcore.h>
#include<cv.h>
#include<highgui.h>
#include<iostream> 
#include<time.h>
#include<string>
using namespace std;
float dotop(float divI[3],float deltaFB[3])
{
	return 2.0f*(divI[0]*deltaFB[0])+4.0f*(divI[1]*deltaFB[1])+3.0f*(divI[2]*deltaFB[2]);
}

void alpha_Initialize(IplImage *alpha,IplImage *cimg,IplImage *foreground,IplImage *background)
{
	if(intest){cout<<endl<<">>>>>alpha_Initialize<<<<<"<<endl;}
	CvScalar alphapixpoint,srcpixpoint,forepixpoint,backpixpoint;
	for(int i=0;i<alpha->height;i++)//這裡的i和j的定義和其他sub function不一樣
	{
		for(int j=0;j<alpha->width;j++)
		{
			alphapixpoint=cvGet2D(alpha,i,j); // get the (i,j) pixel value
			srcpixpoint=cvGet2D(cimg,i,j); // get the (i,j) pixel value
			forepixpoint=cvGet2D(foreground,i,j);
			backpixpoint=cvGet2D(background,i,j);
			
			if(alphapixpoint.val[0]==0.0f)
			{
				alphapixpoint.val[0]=0.0f;
			}
			else if(alphapixpoint.val[0]==1.0f)
			{
				alphapixpoint.val[0]=0.0f;
			}
			else
			{
				float deltaIB[3];
				deltaIB[0]=srcpixpoint.val[0]-backpixpoint.val[0];
				deltaIB[1]=srcpixpoint.val[1]-backpixpoint.val[1];
				deltaIB[2]=srcpixpoint.val[2]-backpixpoint.val[2];
				float deltaFB[3];
				deltaFB[0]=forepixpoint.val[0]-backpixpoint.val[0];
				deltaFB[1]=forepixpoint.val[1]-backpixpoint.val[1];
				deltaFB[2]=forepixpoint.val[2]-backpixpoint.val[2];
				float numerator = dotop(deltaIB,deltaFB);
				float denominator = dotop(deltaFB,deltaFB);
				if (denominator != 0.0f)
					alphapixpoint.val[0] = numerator/denominator;
				else
					alphapixpoint.val[0] = 0.0f;
			}
			cvSet2D(alpha,i,j,alphapixpoint);
		}
	}
}

void alpha_alphaGradient(IplImage *alpha,IplImage *cimg,IplImage *foreground,IplImage *background)
{
	if(intest){cout<<endl<<">>>>>alpha_alphaGradient<<<<<"<<endl;}
	CvScalar alphapixpoint,srcpixpoint,forepixpoint,backpixpoint;
	for(int i=0;i<alpha->height;i++)//這裡的i和j的定義和其他sub function不一樣
	{
		for(int j=0;j<alpha->width;j++)
		{
			alphapixpoint=cvGet2D(alpha,i,j); // get the (i,j) pixel value
			srcpixpoint=cvGet2D(cimg,i,j); // get the (i,j) pixel value
			forepixpoint=cvGet2D(foreground,i,j);
			backpixpoint=cvGet2D(background,i,j);
			if(alphapixpoint.val[0]!=0 && alphapixpoint.val[0]!=255)
			{
				if(i+1<alpha->height && i-1>0 && j+1<alpha->width && j-1>0)
				{
					float divI[3];
					divI[0]=(cvGet2D(cimg,i-1,j).val[0])+(cvGet2D(cimg,i+1,j).val[0])+(cvGet2D(cimg,i,j-1).val[0])+(cvGet2D(cimg,i,j+1).val[0])-4.0f*(cvGet2D(cimg,i,j).val[0]);
					divI[1]=(cvGet2D(cimg,i-1,j).val[1])+(cvGet2D(cimg,i+1,j).val[1])+(cvGet2D(cimg,i,j-1).val[1])+(cvGet2D(cimg,i,j+1).val[1])-4.0f*(cvGet2D(cimg,i,j).val[1]);
					divI[2]=(cvGet2D(cimg,i-1,j).val[2])+(cvGet2D(cimg,i+1,j).val[2])+(cvGet2D(cimg,i,j-1).val[2])+(cvGet2D(cimg,i,j+1).val[2])-4.0f*(cvGet2D(cimg,i,j).val[2]);
					float deltaFB[3];
					deltaFB[0]=forepixpoint.val[0]-backpixpoint.val[0];
					deltaFB[1]=forepixpoint.val[1]-backpixpoint.val[1];
					deltaFB[2]=forepixpoint.val[2]-backpixpoint.val[2];
					alphapixpoint.val[0]=dotop(divI,deltaFB)/(dotop(deltaFB,deltaFB)+0.001f);
				}
			}
			else
			{
				alphapixpoint.val[0]=0.0f;
			}
			cvSet2D(alpha,i,j,alphapixpoint);
		}
	}
}

void foregroungfunction(IplImage *alpha,IplImage *cimg,IplImage *foreground)
{
	if(intest){cout<<endl<<">>>>>foregroungfunction<<<<<"<<endl;}
	//point fill
		CvScalar forepixpoint,alphapixpoint,srcpixpoint;
		for(int i=0;i<alpha->height;i++)
		{
			for(int j=0;j<alpha->width;j++)
			{
				alphapixpoint=cvGet2D(alpha,i,j); // get the (i,j) pixel value
				srcpixpoint=cvGet2D(cimg,i,j); // get the (i,j) pixel value
				forepixpoint=cvGet2D(foreground,i,j);
				if(alphapixpoint.val[0]==255)
				{	
					forepixpoint.val[0]=srcpixpoint.val[0];
					forepixpoint.val[1]=srcpixpoint.val[1];
					forepixpoint.val[2]=srcpixpoint.val[2];
				}
				else
				{
					forepixpoint.val[0]=0.0f;
					forepixpoint.val[1]=0.0f;
					forepixpoint.val[2]=0.0f;
				}
				cvSet2D(foreground,i,j,forepixpoint);//set the (i,j) pixel value
			}
		}
		
		int t0,t1;
	//linear fill,true
	//linear fill,false
	
}

void backgroungfunction(IplImage *alpha,IplImage *cimg,IplImage *background)
{
	if(intest){cout<<endl<<">>>>>backgroungfunction<<<<<"<<endl;}
	//point fill
		CvScalar alphapixpoint,srcpixpoint,backpixpoint;
		for(int i=0;i<alpha->height;i++)
		{
			for(int j=0;j<alpha->width;j++)
			{
				alphapixpoint=cvGet2D(alpha,i,j); // get the (i,j) pixel value
				srcpixpoint=cvGet2D(cimg,i,j); // get the (i,j) pixel value
				backpixpoint=cvGet2D(background,i,j);
				if(alphapixpoint.val[0]==0)
				{	

					backpixpoint.val[0]=srcpixpoint.val[0];
					backpixpoint.val[1]=srcpixpoint.val[1];
					backpixpoint.val[2]=srcpixpoint.val[2];
				}
				else
				{
					backpixpoint.val[0]=255.0f;
					backpixpoint.val[1]=255.0f;
					backpixpoint.val[2]=255.0f;
				}
				cvSet2D(background,i,j,backpixpoint);//set the (i,j) pixel value
			}
		}
	//linear fill,true
	//linear fill,false
}

void poissonProcess(IplImage *timg,IplImage *cimg,IplImage *foreground,IplImage *background)
{
	if(intest){cout<<endl<<">>>>>poissonProcess<<<<<"<<endl;}
//for trimap
	int theight,twidth,tstep,tchannels,*tIp;
	uchar *tdata;
	theight    = timg->height;
	twidth     = timg->width;
	tstep      = timg->widthStep;
	tchannels  = timg->nChannels;
	tdata      = (uchar *)timg->imageData;
	printf("Processing a %dx%d image with %d channels in trimap\n",theight,twidth,tchannels);
	cout<<"(height, width, step, channels)=("<<theight<<", "<<twidth<<", "<<tstep<<", "<<tchannels<<")"<<endl;
//for color image data
	int height,width,step,channels;
	uchar *data;
	height    = cimg->height;
	width     = cimg->width;
	step      = cimg->widthStep;
	channels  = cimg->nChannels;
	data      = (uchar *)cimg->imageData;
	printf("Processing a %dx%d image with %d channels in color image\n",height,width,channels);
	cout<<"(height, width, step, channels)=("<<height<<", "<<width<<", "<<step<<", "<<channels<<")"<<endl;
//alphaFromTrimap
	IplImage *alpha = cvCreateImage(cvGetSize(timg),timg->depth,timg->nChannels);
	cvCopy(timg,alpha, NULL);

//calculate foreground
	foregroungfunction(alpha,cimg,foreground);
//calculate backgroung
	backgroungfunction(alpha,cimg,background);
//alpha calculate
	alpha_alphaGradient(alpha,cimg,foreground,background);
	alpha_Initialize(alpha,cimg,foreground,background);
/*	cvNamedWindow("alpha",CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("alpha",800,100);
	cvShowImage("alpha", alpha ); 
	cvWaitKey(0); 
	cvReleaseImage(&alpha);*/
}

int main()
{
	if(intest){cout<<">>>>>main<<<<<"<<endl;}
	//initial timing analysis
	clock_t start,end;
	start=clock();
	//initial image import
	IplImage *cimg,*timg;//,*gimg;
	IplImage *foreground, *background;
	//user input
	cout<<"Enter input image : "<<endl;
	string inputcolor;
	if(intest)
	{
		inputcolor="test.png";
		cout<<inputcolor<<endl;
	}
	else
		cin>>inputcolor;
	cout<<"Enter trimap : "<<endl;
	string inputtrimap;
	if(intest)
	{
		inputtrimap="test_tri.png";
		cout<<inputtrimap<<endl;
	}
	else
		cin>>inputtrimap;
	// load an image   
	cimg=cvLoadImage(inputcolor.c_str(),-1);
	timg=cvLoadImage(inputtrimap.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
	foreground=cvLoadImage(inputcolor.c_str(),-1);;
	background=cvLoadImage(inputcolor.c_str(),-1);;
	// create a window
	cvNamedWindow("original color image", CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("original color image" , 700, 100); 
	cvNamedWindow("trimap"				,CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("trimap"				, 700, 100);  
	cvNamedWindow("foreground"			,CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("foreground"			, 700, 100); 
	cvNamedWindow("background"			,CV_WINDOW_AUTOSIZE); 
	cvMoveWindow("background"			, 700, 100);
	//image process
	poissonProcess(timg,cimg,foreground,background);

	// show the image
	cvShowImage("original color image", cimg ); 
	cvShowImage("trimap", timg ); 
	cvShowImage("foreground", foreground ); 
	cvShowImage("background", background ); 
	//timing analysis
	end=clock();
	cout<<endl<<"rumtime="<<((double)(end-start))/((double)CLOCKS_PER_SEC)*1000<<"(ms)="<<((double)(end-start))/((double)CLOCKS_PER_SEC)<<"(sec.)"<<endl;
	// wait for a key
	cvWaitKey(0); 
 
	// release the image
	cvReleaseImage(&cimg );
	cvReleaseImage(&timg );
	cvReleaseImage(&foreground);
	cvReleaseImage(&background);
	return 0;
}

