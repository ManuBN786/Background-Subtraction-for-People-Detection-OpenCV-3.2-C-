#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
#include <cstring>
#include <ctime>


#include "opencv2/video/background_segm.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


   

using namespace cv;
using namespace std;
RNG rng(12345);


// Trial to fill holes
void cvFillHoles(cv::Mat &input)
{
    //assume input is uint8 B & W (0 or 1)
    //this function imitates imfill(image,'hole')
    cv::Mat holes=input.clone();
    cv::floodFill(holes,cv::Point2i(0,0),cv::Scalar(1));
    for(int i=0;i<input.rows*input.cols;i++)
    {
        if(holes.data[i]==0)
            input.data[i]=1;
    }
}

 // plot points
                              #define drawCross( p1, color, d )                                        \
                              line( img, Point( p1.x - d, p1.y - d ),                          \
                                Point( p1.x + d, p1.y + d ), color, 1, LINE_AA, 0); \
                              line( img, Point( p1.x + d, p1.y - d ),                          \
                                Point( p1.x - d, p1.y + d ), color, 1, LINE_AA, 0 )

// To calculate the center points
static inline Point calcPoint(Point2f center, double R, double angle)
     {
        return center + Point2f((float)cos(angle), (float)-sin(angle))*(float)R;
    }

//int LeftExit = 0, RightExit = 0;
//int LeftEntry = 0, RightEntry = 0;

 int LeftExit = 0, RightExit = 0;
                            int LeftEntry = 0, RightEntry = 0;

using namespace cv;


int main(int argc, const char** argv)
{


	// Init background substractor
	Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();

	// Create empty input img, foreground and background image and foreground mask.
	Mat img, foregroundMask, backgroundImage, foregroundImg;



	// capture video from source 0, which is web camera, If you want capture video from file just replace //by
	//VideoCapture cap("20160624_130000_CH1_0.avi");
	VideoCapture cap(0);

       // Vectors to store the tracking points of actual and kalman predicted values.
         vector<Point> actualv,kalmanv;
         actualv.clear(); // Clear both the vectors initially
         kalmanv.clear();

        // Vectors for counting Entry & Exit
         vector<int>LeftExitV;
         vector<int>RightExitV;
         vector<int>LeftEntryV;
         vector<int>RightEntryV;

          LeftExitV.clear();               // Clear all the vectors
          RightExitV.clear();
          LeftEntryV.clear();
          RightEntryV.clear();




	// main loop to grab sequence of input files
	for (;;){


		bool ok = cap.grab();

		if (ok == false){

			std::cout << "Video Capture Fail" << std::endl;


		}
		else{

			// obtain input image from source
			cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
			// Just resize input image if you want
			resize(img, img, Size(640, 480));

			// create foreground mask of proper size
			if (foregroundMask.empty()){
				foregroundMask.create(img.size(), img.type());
			}

			// compute foreground mask 8 bit image
			// -1 is parameter that chose automatically your learning rate

			bg_model->apply(img, foregroundMask, true ? -1 : 0);

			// smooth the mask to reduce noise in image
			GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);

			// threshold mask to saturate at black and white values
			threshold(foregroundMask, foregroundMask, 50, 255, THRESH_BINARY);

			// Invert Foreground Mask
			//bitwise_not(foregroundMask, foregroundMask);

                        
                        // Morphological Operations on Foreground Mask
                         //morphologyEx(foregroundMask,foregroundMask,MORPH_OPEN,getStructuringElement(MORPH_RECT,Size(4,4)),Point(3,3));
                         dilate( foregroundMask, foregroundMask,getStructuringElement(MORPH_RECT,Size(25,25)),Point(3,3));
                        morphologyEx(foregroundMask,foregroundMask,MORPH_CLOSE,getStructuringElement(MORPH_RECT,Size(25,25)),Point(3,3));
                        imshow( "Morphology", foregroundMask );
                       

                       // Imfill operations to fill with holes use cvFillHoles
                        // Threshold.
                       // Set values equal to or above 220 to 0.
                      // Set values below 220 to 255.
                   //  Mat im_th;
                     // threshold(foregroundMask, im_th, 50, 255, THRESH_BINARY_INV);
                    //  Floodfill from point (0, 0)
                    // Mat im_floodfill = foregroundMask.clone();
                    // floodFill(im_floodfill, cv::Point(0,0), Scalar(255));
     
                      // Invert floodfilled image
                   //  Mat im_floodfill_inv;
                   //  bitwise_not(im_floodfill, im_floodfill_inv);
                    //  foregroundMask = (foregroundMask | im_floodfill_inv);

                    cvFillHoles(foregroundMask);
                    	// imshow( "Holes filled", foregroundMask );

			//Background Subtractor


				// create black foreground image
				foregroundImg = Scalar::all(0);
			// Copy source image to foreground image only in area with white mask
			img.copyTo(foregroundImg, foregroundMask);

			//Get background image
			bg_model->getBackgroundImage(backgroundImage);

                        // Find Contours
                        int thresh = 100;
                        int max_thresh = 255;
                        RNG rng(12345);
                        Mat canny_output;
                        vector<vector<Point> > contours;
                        //vector<Rect2d<Point>> contours;
                        vector<Vec4i> hierarchy;
                        Canny( foregroundMask, canny_output, thresh, thresh*2, 3 );
                        findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
                        Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );

                        // Initialize the no of contours to zero
                        //int num_contour = 0;
                        
                        Rect bounding_rect;


                        // Draw the contours, use false to identify incomplete shapes i.e not closed
                        for( size_t i = 0; i< contours.size(); i++ )
                                 {
                                   //if (contourArea( contours[i],true) < 50000)
                                     if (arcLength( contours[i],false) < 850)
                                      continue;
                                    // num_contour++;
                                   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
                                   drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
                                   bounding_rect=boundingRect(contours[i]);
                                  rectangle(img, bounding_rect,  Scalar(255,0,0),1, 8,0);
                                 }

                       //cout<<" No. of people are "<<num_contour; 
                       imshow( "Contours", drawing );               
                       imshow( "Contours Plotted", img );



                        //rectangle(foregroundMask, bounding_rect,  Scalar(0,255,0),1, 8,0);
			// Show the results
		//	imshow("foreground mask", foregroundMask);

		//	imshow("foreground image", foregroundImg);

			// Setup SimpleBlobDetector parameters.
		//	SimpleBlobDetector::Params params;

			// Filter by Convexity
			//params.filterByConvexity = true;
			//params.minConvexity = 0.6;
			//maxConvexity = 1;

			// Change thresholds
			//params.minThreshold = 20;
			//params.maxThreshold = 200;

			// Filter by Area.
			//params.filterByArea = true;
			//params.minArea = 5800;
			//params.maxArea = 1000;

		//	params.minThreshold = 40;
		//	params.maxThreshold = 60;
		//	params.thresholdStep = 5;
			//params.minArea = 7000;
		//	params.minArea = 40000;
		//	params.minConvexity = 0.3;
		//	params.minInertiaRatio = 0.01;
		//	params.maxArea = 250000;
			//params.maxArea = 120000;
		//	params.maxConvexity = 10;

			// Storage for blobs
		//	vector<KeyPoint> keypoints;
			// Set up detector with params
		//	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

			//bitwise_not(foregroundImg, foregroundImg);
                        // Morphological Operations
                        //morphologyEx(foregroundImg,foregroundImg,MORPH_OPEN,getStructuringElement(MORPH_RECT,Size(5,5)));
                       // morphologyEx(foregroundImg,foregroundImg,MORPH_CLOSE,getStructuringElement(MORPH_RECT,Size(15,15)));

			// Detect blobs
		//	detector->detect(foregroundImg, keypoints);
			


			// Draw detected blobs as red circles.
			// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
			// the size of the circle corresponds to the size of blob

		//	Mat im_with_keypoints;
		//	drawKeypoints(foregroundImg, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			//imshow("Keypoints image", im_with_keypoints);
                       // cout<<keypoints<<endl;




			// Find and display the no. of keypoints i.e no. of blobs
		//	int numBlobs = keypoints.size();
			//cout << "No. of People are " << numBlobs << endl;
                        
                        
                        // Find the centroid of blobs
                        //float x_cor = keypoints.pt.x;
                  //      float sumx=0, sumy=0;
                 //       float num_pixel = 0;
                 //       for(int x=0; x<foregroundMask.cols; x++) {
                //        for(int y=0; y<foregroundMask.rows; y++) {
                //        int val = foregroundMask.at<uchar>(y,x);
                        // Arbitrary threshold
               //         if( val >= 50) {
               //         sumx += x;
               //         sumy += y;
               //         num_pixel++;
                //                       }
                //                      }
                //                     }
                       // Point p(sumx/num_pixel, sumy/num_pixel);
                        // p is the centroid evaluated manually
                        //cout << Mat(p) << endl;

                       // Moments m = moments(foregroundMask, false);
                       // Point p1(m.m10/m.m00, m.m01/m.m00);
                        // p1 is the centroid evaluated by moments
                        //cout<<p1<<endl;
                        //cout << "X Co-ordinates"<<p1.x << endl;
                       // cout << "Y Co-ordinates"<<p1.y << endl;

                        
                       // circle(foregroundMask, p1, 5, Scalar(128,0,0), -1);
                        //imshow("Centroid Found", foregroundMask);
                     
                        int num_contour = 0;
                        // Get the moments 
                        vector<Moments> m(contours.size() );
                        vector<Point2f> mc( m.size() );
                        for( size_t j = 0; j < contours.size(); j++ )
                        {   

                        if (arcLength( contours[j],false) > 850)
                           num_contour++;
                           m[j] = moments( contours[j], false ); 
                           mc[j] = Point2f( m[j].m10/m[j].m00 , m[j].m01/m[j].m00);  // Centroid of the contour

                           cout<<" the no of people are "<< num_contour<<endl;
                           cout<<mc[j]<<endl;


                         // the fixed co ordinates of the room w.r.t beam
                             Point L1_1 = Point(63,1);
                             Point L1_2 = Point(63,480);

                            Point L2_1 = Point(577,1);
                            Point L2_2 = Point(577,480);



                            line(img,L1_1,L1_2,Scalar(0,0,255),3,8);
                            line(img,L2_1,L2_2,Scalar(0,0,255),3,8); 


                             Point2f Left =  Point(63,226);
                             Point2f Right = Point(577,226);

                             drawCross(Left, Scalar(0,0,255), 3 );
                             drawCross(Right, Scalar(0,0,255),3);
                           //drawCross(mc[j],Scalar(255,0,255),3);

                            //Initialize entry & Exits

                           // int LeftExit = 0, RightExit = 0;
                           // int LeftEntry = 0, RightEntry = 0;


//if (mc[j].x < Left.x)
 //  LeftExit++;
//cout<<"Left_Exit = "<<LeftExit<<endl;

//if (mc[j].x > Right.x)
//   RightExit++;
//cout<<"Right_Exit = "<<RightExit<<endl;

                            if (mc[j].x < Left.x)
                            RightExit++;
                            cout<<"Right_Exit = "<<RightExit<<endl;

                            if (mc[j].x > Right.x)
                            LeftExit++;
                            cout<<"Left_Exit = "<<LeftExit<<endl;


                           // Right entry
                           if ((mc[j-1].x > Right.x)&&(mc[j].x< Right.x))
                              //&&(mc[j].x>Left.x))
                           LeftEntry++;
                           cout<<"Left_Entry = "<<LeftEntry<<endl;

                          // Left entry
                          if ((mc[j-1].x < Left.x)&&(mc[j].x > Left.x))
                          //&&(mc[j].x < Right.x))
                            RightEntry++;
                          cout<<"Right_Entry = "<<RightEntry<<endl;


                           LeftExitV.push_back(LeftExit);
                           RightExitV.push_back(RightExit);
                           LeftEntryV.push_back(LeftEntry);
                           RightEntryV.push_back(RightEntry);

                          //cout<<LeftExitV.size()<<endl;
//cout<<LeftExitV.size()<<endl;

//int RightDisplay = 0;

//for (Point2f u = 1; u < LeftExit.size(); u++){
//    if (LeftExit(u) == 1)
//    RightDisplay++;
// }

//int RightDisplay = LeftExitV.size();
//cout<<RightDisplay<<endl; 
                          imshow( "Crosses", img );









                         // Kalman Filter Starts Here
                         KalmanFilter KF(2, 1, 0);
                         Mat state(2, 1, CV_32F); /* (phi, delta_phi) */
                         Mat processNoise(2, 1, CV_32F);
                         Mat measurement = Mat::zeros(1, 1, CV_32F);
                         char code = (char)-1;

                         

//  Get the mass centers:
     // vector<Point2f> mc( mu.size() );
      //for( size_t i = 0; i < mu.size(); i++ ) 

      
       //   mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00); 

      
                       

                         
                        // for(;;)
                        // {
                              randn( state, Scalar::all(0), Scalar::all(0.1) );
                              KF.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);

                            //  KF.statePre.at<float>(0) = p1.x;
                            //  KF.statePre.at<float>(1) = p1.y;
    
                              setIdentity(KF.measurementMatrix);
                              setIdentity(KF.processNoiseCov, Scalar::all(1e-5));
                              setIdentity(KF.measurementNoiseCov, Scalar::all(1e-1));
                              setIdentity(KF.errorCovPost, Scalar::all(1));
    
                              randn(KF.statePost, Scalar::all(0), Scalar::all(0.1));
                             //  for(;;)
                             //    {
                                   //   Point2f center(im_with_keypoints.cols*0.5f, im_with_keypoints.rows*0.5f);
                                      float R = img.cols/1.f;
                                      double stateAngle = state.at<float>(0);
                                      Point statePt[j] = calcPoint(mc[j],R,stateAngle);

                                      Mat prediction = KF.predict();
                                      double predictAngle = prediction.at<float>(0);
                                      Point predictPt[j] = calcPoint(mc[j], R, predictAngle);
                                      randn( measurement, Scalar::all(0), Scalar::all(KF.measurementNoiseCov.at<float>(0)));

                             // generate measurement
                               measurement += KF.measurementMatrix*state;
   
                               double measAngle = measurement.at<float>(0);
                                     Point measPt[j] = calcPoint(mc[j], R, measAngle);
   
                            


                               //img = Scalar::all(0);
                               drawCross( statePt[j], Scalar(255,255,255), 3 );
                               drawCross( measPt[j], Scalar(0,0,255), 3 );
                               drawCross( predictPt[j], Scalar(0,255,0), 3 );
                               line( img, statePt[j], measPt[j], Scalar(0,0,255), 3, LINE_AA, 0 );
                               line( img, statePt[j], predictPt[j], Scalar(0,255,255), 3, LINE_AA, 0 );
    
                               if(theRNG().uniform(0,4) != 0)
                               KF.correct(measurement);
    
                               randn( processNoise, Scalar(0), Scalar::all(sqrt(KF.processNoiseCov.at<float>(0, 0))));
                               state = KF.transitionMatrix*state + processNoise;


// To trace
                               
                             actualv.push_back(measPt[j]);
                             kalmanv.push_back(statePt[j]);

                             for (int k = 0; k < actualv.size()-1; k++) 
                             line(img, actualv[k], actualv[k+1], Scalar(0,0,255), 1, LINE_AA,0);
     
                             for (int k = 0; k < kalmanv.size()-1; k++) 
                             line(img, kalmanv[k], kalmanv[k+1], Scalar(0,255,255), 1,LINE_AA,0);


                              // imshow( "Kalman", img );
                            }
                     //drawKeypoints(foregroundImg, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);


                               // To trace
                               
                              // actualv.push_back(measPt);
                              // kalmanv.push_back(statePt);

                             // for (int i = 0; i < actualv.size()-1; i++) 
                             // line(im_with_keypoints, actualv[i], actualv[i+1], Scalar(0,0,255), 1, LINE_AA,0);
     
                             //for (int i = 0; i < kalmanv.size()-1; i++) 
                            //  line(im_with_keypoints, kalmanv[i], kalmanv[i+1], Scalar(0,255,255), 1,LINE_AA,0);

                              
                           //    code = (char)waitKey(100);
    
                               // Press Keys to Stop Simulation
                           //    if( code > 0 )
                           //    break;
                              //    }
                            //   if( code == 27 || code == 'q' || code == 'Q' )
                            //   break;
                       //   }

                  



// Multi tracker from opencv 3.1
    
// set the default tracking algorithm
 //std::string trackerType = "KCF";
  
 //Ptr<Tracker> tracker = Tracker::create( "KCF" );

 // set the tracking algorithm from parameter
// if(argc>2)
// trackingAlg = argv[2];
 
  // create the tracker
 // MultiTracker trackers(tracker);
  // get bounding box

 // selectROI("tracker",img,contours);


//waitKey(100);

                      // To display the background image

                      int key6 = waitKey(40);

			//if (!backgroundImage.empty()){

				//imshow("mean background image", backgroundImage);

			//	int key5 = waitKey(40);

			//}


		}

	}

	return EXIT_SUCCESS;
}
