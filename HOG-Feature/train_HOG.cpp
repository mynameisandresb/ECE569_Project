#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <time.h>

using namespace cv;
using namespace cv::ml;
using namespace std;

void get_svm_detector( const Ptr< SVM > & svm, vector< float > & hog_detector );
void convert_to_ml( const std::vector< Mat > & train_samples, Mat& trainData );
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages );
void load_actual_images(vector< Mat > & img_lst, bool showImages );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip );
int test_trained_detector( String obj_det_filename, String test_dir, String videofilename );
void combinedTrainingData(const vector< Mat > & img_lst, vector< Mat > & gradient_lst );

void get_svm_detector( const Ptr< SVM >& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml( const vector< Mat > & train_samples, Mat& trainData )
{
    //Convert data
    const int rows = (int)train_samples.size();
    const int cols = train_samples[0].cols * train_samples[0].rows;
    Mat tmp( 1, cols, CV_32F ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32F );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        clog << train_samples[i].cols << "\n"<< train_samples[i].rows << " \n\n";

        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

//CV Matrix and store in vector structure 
void load_images( const String & dirname, vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    glob( dirname, files );
    for ( size_t i = 0; i < files.size(); ++i )
    {
        FileStorage fs(files[i], FileStorage::READ);
        // Load the matrix from the file
        Mat A;
        fs["data"] >> A;

        fs.release();

        cout << files[i] << endl;
        //cout << "A = " << endl << " "  << A << endl << endl;
        Size pos_image_size = A.size();
        cout << "Size:" << A.size() << endl;

        img_lst.push_back( A );
        //A.release();
    }
}

//Sample negative vector array, and resize if it does not match the positive  
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
    Rect box;
    box.width = size.width;
    box.height = size.height;

    const int size_x = box.width;
    const int size_y = box.height;
    clog << "sizex.."<< size_x ;
    
    Mat B = Mat::ones(size,CV_32F); 

    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
    {
        
        int t_rows = size_y-full_neg_lst[i].size().height;        
        Mat dw = Mat::zeros(t_rows,full_neg_lst[i].cols,full_neg_lst[i].type());

        Mat c; 
        cv::vconcat(full_neg_lst[i], dw,c);
       
        neg_lst.push_back( c.clone() );
    }
 
}

void computeHOGs( const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst, bool use_flip )
{
    HOGDescriptor hog;
    hog.winSize = wsize;
    Mat gray;
    vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ )
    {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height )
        {
            Rect r = Rect(( img_lst[i].cols - wsize.width ) / 2,
                          ( img_lst[i].rows - wsize.height ) / 2,
                          wsize.width,
                          wsize.height);
            cvtColor( img_lst[i](r), gray, COLOR_BGR2GRAY );
            hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
            gradient_lst.push_back( Mat( descriptors ).clone() );
            if ( use_flip )
            {
                flip( gray, gray, 1 );
                hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ) );
                gradient_lst.push_back( Mat( descriptors ).clone() );
            }
        }
    }
}

void combinedTrainingData(const vector< Mat > & img_lst, vector< Mat > & gradient_lst ){
    for( size_t i=0 ; i< img_lst.size(); i++ )
    {
        gradient_lst.push_back( img_lst[i].clone() );
    }
}


int test_trained_detector( String obj_det_filename, String test_dir, String videofilename )
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );

    vector< String > files;
    glob( test_dir, files );

    int delay = 0;
    VideoCapture cap;

    if ( videofilename != "" )
    {
        if (videofilename.size() == 1 && isdigit(videofilename[0]))
            cap.open(videofilename[0] - '0');
        else
        cap.open( videofilename );
    }

    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );

    for( size_t i=0;; i++ )
    {
        Mat img;

        if ( cap.isOpened() )
        {
            cap >> img;
            delay = 1;
        }
        else if( i < files.size() )
        {
            img = imread( files[i] );
        }

        if ( img.empty() )
        {
            return 0;
        }

        vector< Rect > detections;
        vector< double > foundWeights;

        hog.detectMultiScale( img, detections, foundWeights );
        for ( size_t j = 0; j < detections.size(); j++ )
        {
            Scalar color = Scalar( 0, foundWeights[j] * foundWeights[j] * 200, 0 );
            rectangle( img, detections[j], color, img.cols / 400 + 1 );
        }

        imshow( obj_det_filename, img );

        if( 27 == waitKey( delay ) )
        {
            return 0;
        }
    }
    return 0;
}

void load_actual_images( vector< Mat > & img_lst, bool showImages = false )
{
    vector< String > files;
    cout << " whhaaa \n" << " \n";
    glob( "../video_converter/images", files );
    cout << files.size() << " \n";
    for ( size_t i = 0; i < files.size(); ++i )
    {
        Mat img = imread( files[i] ); // load the image
        if ( img.empty() )
        {
            cout << files[i] << " is invalid!" << endl; // invalid image, skip it.
            continue;
        }
        if ( showImages )
        {
            imshow( "image", img );
            waitKey( 1 );
        }
        img_lst.push_back( img );
    }
}

int main( int argc, char** argv )
{
    
    const char* keys =
    {
        "{help h|     | show help message}"
        "{pd    |     | path of directory contains possitive images}"
        "{nd    |     | path of directory contains negative images}"
        "{td    |     | path of directory contains test images}"
        "{tv    |     | test video file name}"
        "{dw    |     | width of the detector}"
        "{dh    |     | height of the detector}"
        "{d     |false| train twice}"
        "{t     |false| test a trained detector}"
        "{v     |false| visualize training steps}"
        "{fn    |my_detector.yml| file name of trained SVM}"
    };

    CommandLineParser parser( argc, argv, keys );

    if ( parser.has( "help" ) )
    {
        parser.printMessage();
        exit( 0 );
    }

    String pos_dir = parser.get< String >( "pd" );
    String neg_dir = parser.get< String >( "nd" );
    String test_dir = parser.get< String >( "td" );
    String obj_det_filename = parser.get< String >( "fn" );
    String videofilename = parser.get< String >( "tv" );
    int detector_width = parser.get< int >( "dw" );
    int detector_height = parser.get< int >( "dh" );
    bool test_detector = parser.get< bool >( "t" );
    bool train_twice = parser.get< bool >( "d" );
    bool visualization = parser.get< bool >( "v" );

    if ( test_detector )
    {
        test_trained_detector( obj_det_filename, test_dir, videofilename );
        exit( 0 );
    }

    //hardcode 
    pos_dir = "../video_converter/1uofa";
    neg_dir = "../video_converter/1asu";

    vector< Mat > pos_lst, full_neg_lst, gradient_lst;
    vector< int > labels;

    // Load images
    clog << "Positive images are being loaded...\n" ;
    load_images( pos_dir, pos_lst, false );
    for ( size_t i = 0; i < pos_lst.size(); ++i )
    {
        gradient_lst.push_back(pos_lst[i]);
    }
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    clog << "...[done] ( positive images count : " << positive_count << " )" << endl;

    //size of input data
    Size pos_image_size = pos_lst[0].size();
    Mat train_data;
    const unsigned int old = (unsigned int)labels.size();

    // Load negative images
    load_images( neg_dir, full_neg_lst, false );
    for ( size_t i = 0; i < full_neg_lst.size(); ++i )
    {
        gradient_lst.push_back(full_neg_lst[i]);
    }
    clog << "Negative images are  loaded...";

    //Add label to ngatives
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, 0 );
    clog << "labels.insert.. Check..";
    CV_Assert( old < labels.size() );
  
    convert_to_ml( gradient_lst, train_data );
    clog << pos_lst.size() << "\n\n";
    clog << full_neg_lst.size() << "\n\n";
    clog << gradient_lst.size() << "\n\n";
    clog << train_data.size() << "\n\n";
    clog << labels.size() << "\n\n";
    clog << "completed ...";


    clog << "Training SVM...";
    Ptr< SVM > svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    clog << "completed1 ...";
    svm->setDegree( 3 );
    clog << "completed2 ...";
    svm->setTermCriteria( TermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3 ) );
    clog << "completed3 ...";
    svm->setGamma( 0 );
    clog << "completed4 ...";
    svm->setKernel( SVM::LINEAR );
    clog << "completed5 ...";
    svm->setNu( 0.5 );
    clog << "completed6 ...";
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    clog << "completed7 ...";
    svm->setC( 0.01 ); // From paper, soft classifier
    clog << "completed8 ...";
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    clog << "completed9 ...";
    svm->train( train_data, ROW_SAMPLE, Mat( labels ) );
    clog << "...[done]" << endl;

    // clog << "After train 1\n" << endl;
    // vector< float > hog_detector;
    // clog << "After train 2\n" << endl;
    // get_svm_detector( svm, hog_detector );
    // clog << "After train 3\n" << endl;
    svm->save("svm.yml");

    // HOGDescriptor hog;
    // clog << "After train 4\n" << endl;
    // hog.winSize = pos_image_size;
    // clog << "After train 5\n" << endl;
    // hog.setSVMDetector( hog_detector );
    // clog << "After train 6\n" << endl;
    // hog.save( obj_det_filename );
    // clog << "After train 7\n" << endl;
    // test_trained_detector( obj_det_filename, test_dir, videofilename );
    // clog << "After train 8\n" << endl;
    return 0;
}