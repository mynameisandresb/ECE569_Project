/*
* Code:     train_HOG.cpp 4/25/23
* Purpose:  
*   This project creates an SVM to classify two object based on their feature data 
* Setup:
*   GCC V4.8.5 and OpenCV 3.3.0
*   1x directory with Positive Object HOG Feature Data at various 
*   1x directory with Negative Object HOG Feature Data 
*   Feature Data must be save as a CV Matrix *.yml with "data" identifier
*   Feature Data for both objects must have the same histogram configure
*   (i.e. bin size, cellsize, blocksize, etc...)
* Build:
*   $ g++ train_HOG.cpp -o train_HOG `pkg-config --cflags --libs opencv`
*   $ ./train_HOG -pd=<positive data path> -nd=<negative data path> -fn=<output SVM File>
*/

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
void load_feature_data( const String & dirname, vector< Mat > & data_lst);
int test_trained_detector( String obj_det_filename, String test_dir);



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

/*
* Load feature matrices data, and store in vector structure
* The Vector is a 1D array of CV Matrix data
*/ 
void load_feature_data( const String & dirname, vector< Mat > & data_lst)
{
    //Create Temp vector to account for files
    vector< String > files;
    glob( dirname, files );
    //Parse data out of each file
    for ( size_t i = 0; i < files.size(); ++i )
    {
        //Read in each feature file
        FileStorage fs(files[i], FileStorage::READ);
        //Load the matrix from the file, grad "data" labeled structure
        Mat A;                 
        //Store data in CV Matrix 
        fs["data"] >> A;    
        //Release File    
        fs.release();          
        //Append Matrix A to global vector data_lst 
        data_lst.push_back( A );
    }
}

/*
* Test a sample Feature CV Matrix File   
* Against SVM 
*/ 
int test_trained_detector( String obj_det_filename, String test_dir)
{
    cout << "Testing trained detector..." << endl;
    HOGDescriptor hog;
    hog.load( obj_det_filename );

    vector< String > files;
    glob( test_dir, files );

    int delay = 0;

    obj_det_filename = "testing " + obj_det_filename;
    namedWindow( obj_det_filename, WINDOW_NORMAL );

    for( size_t i=0;; i++ )
    {
        Mat img;

        if( i < files.size() )
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


int main( int argc, char** argv )
{
    
    const char* keys =
    {
        "{help h|     | show help message}"
        "{pd    |     | path of directory contains possitive features}"
        "{nd    |     | path of directory contains negative features}"
        "{td    |     | path of directory contains test features}"
        "{t     |false| test a trained detector}"
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
    bool test_detector = parser.get< bool >( "t" );
    vector< Mat > pos_lst, full_neg_lst, gradient_lst;
    vector< int > labels;

    if ( test_detector )
    {
        test_trained_detector( obj_det_filename, test_dir);
        exit( 0 );
    }

    if( pos_dir.empty() || neg_dir.empty() )
    {
        parser.printMessage();
        cout << "Wrong number of parameters.\n\n"
             << "Example command line:\n" << argv[0] << " -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian64x128.yml \n"
             << "\nExample command line for testing trained detector:\n" << argv[0] << " -t -fn=HOGpedestrian64x128.xml -td=/INRIAPerson/Test/pos";
        exit( 1 );
    }

    if ( pos_lst.size() > 0 )
    {
        clog << "...[done] " << pos_lst.size() << " files." << endl;
    }
    else
    {
        clog << "no image in " << pos_dir <<endl;
        return 1;
    }

    // Load Load feature data 
    clog << "Positive feature data are being loaded...\n" ;
    load_feature_data( pos_dir, pos_lst);
    for ( size_t i = 0; i < pos_lst.size(); ++i )
    {
        //Append data gradient vector
        gradient_lst.push_back(pos_lst[i]);
    }

    //Add +1 as the lable for positive data 
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    clog << "...[done] ( positive features count : " << positive_count << " )" << endl;

    //create old variable for length of labels for tracking later 
    const unsigned int old = (unsigned int)labels.size();

    // Load negative features
    load_feature_data( neg_dir, full_neg_lst);
    for ( size_t i = 0; i < full_neg_lst.size(); ++i )
    {
        //Append data to gradient vector
        gradient_lst.push_back(full_neg_lst[i]);
    }
    clog << "Negative features are  loaded...";

    //Add negative labels to label vector
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, 0 );
    clog << "labels.insert.. Check..";

    //Confirm old label vector size is NOT greater than the old
    CV_Assert( old < labels.size() );
  
    //Create CV Train Matrix for train_data
    Mat train_data;

    //Convert gradient data to a training data set 
    convert_to_ml( gradient_lst, train_data );
  
   //Print out data sizes for tracking 
    clog << pos_lst.size() << "\n\n";
    clog << full_neg_lst.size() << "\n\n";
    clog << gradient_lst.size() << "\n\n";
    clog << train_data.size() << "\n\n";
    clog << labels.size() << "\n\n";
    clog << "completed ...";

    //Running Training Algorithm for SVM
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
    //Save SVM to Output File yml.
    if(obj_det_filename == ""){
        svm->save("svm.yml");
    }else{
        svm->save(obj_det_filename);
    }

    return 0;
}
