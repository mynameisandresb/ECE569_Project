//###################################################################################################################################################
//------------------------------------------------------------HOG-Feature Section--------------------------------------------------------------------------
//###################################################################################################################################################

//------------------------------------------------------------Includes--------------------------------------------------------------------------
#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string.h>

// CUDA stuff:
#include "cuda_runtime.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


struct HogProp
{
	int ImgRow,ImgCol;
	int CellRow,CellCol;
  int TotalCells;
  int BlockRow,BlockCol;
  int TotalBlocks;
  int FeatureSize;
  int ImgSize;
	int CellSize;
	int BlockSize;
	int BlockOverlap;
	int NumBins;
	int Orientation;
};

struct DisplayProp
{
	int ImgRow,ImgCol;
	int CellRow,CellCol;
  int TotalCells;
  int HorzCells;
  int HorzCellsTotal;
  //int BlockRow,BlockCol;
  //int TotalBlocks;
  //int FeatureSize;
  int ImgSize;
	int CellSize;
	//int BlockSize;
	//int BlockOverlap;
	int NumBins;
  int DisplayCellSize;
  int DisplayImgRow,DisplayImgCol,DisplayImgSize;
	//int Orientation;
};

#define PI 3.141592654
//extern struct HogProp hp;

#define BOX_SIZE 8
cudaError_t launch_helper(float* Runtimes, float *CPU_FeatureArray, uchar *CPU_InputArray, DisplayProp dp, HogProp hp);

using namespace cv;

int Cal_kernel_v;
int Cell_kernel_v;
int Block_kernel_v;
//int Display_Cell_kernel_v;
//int display_kernel_v;

//-------------------------------------------------------------Cal_kernel-------------------------------------------------------------------------

// Cal_kernel Original Version 0
__global__ void Cal_kernel_v0(uchar *GPU_i, int *Orientation,float *Gradient, uchar *DisplayOrientation, HogProp hp){
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  float ang,displayang;
	float gx,gy;
  int idx = i*hp.ImgCol + j;
  int idx_prev= (i-1)*hp.ImgCol + j;
  int idx_next= (i+1)*hp.ImgCol + j;
 
  if(i>0 && i < hp.ImgRow-1 && j >0 && j < hp.ImgCol-1){
   	gx=(float)(GPU_i[idx-1]-GPU_i[idx+1]);
    gy=(float)(GPU_i[idx_prev]-GPU_i[idx_next]);
    
     Gradient[idx]=sqrtf(gx*gx+gy*gy);
     ang= atan2f(gy,gx);
     
     if(ang<0) {
       displayang=8*(ang+PI);
     }
     else displayang=8*ang;
     
     if(displayang<PI | displayang>7*PI)          DisplayOrientation[idx]=0;
     else if(displayang>=PI & displayang<3*PI)    DisplayOrientation[idx]=1;
     else if(displayang>=3*PI & displayang<5*PI)  DisplayOrientation[idx]=2;
     else                                         DisplayOrientation[idx]=3;
          
     if (ang<0){
       if(hp.Orientation==0) { ang = ang+ PI; }
       else { ang = 2*PI+ang; }
     }
     
     if(hp.Orientation==0) ang=(hp.NumBins)*ang/PI;
     else ang=(hp.NumBins)*ang/(2*PI);
     
     Orientation[idx]=(int)ang;
     //GPU_o[idx] = (uchar) (DisplayOrientation[idx]);
  }
}

// Cal_kernel Optimized Version 1
// No significant execution time performance gains
// Replaced the C-style casting with static_cast to improve code safety and readability
// Merged the two separate branches for updating the displayang variable into a single line using the ternary operator, simplifying the control flow
__global__ void Cal_kernel_v1(uchar *GPU_i, int *Orientation,float *Gradient, uchar *DisplayOrientation, HogProp hp){
  // Calculate row and column indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col of image

  // Declare variables for gradient angle, display angle, and gradient components
  float ang, displayang;
  float gx, gy;
  // Calculate linear index for current, previous, and next rows in the image
  int idx = i * hp.ImgCol + j;
  int idx_prev = (i - 1) * hp.ImgCol + j;
  int idx_next = (i + 1) * hp.ImgCol + j;

  // Ensure the current thread is within image boundaries
  if (i > 0 && i < hp.ImgRow - 1 && j > 0 && j < hp.ImgCol - 1){
    // Compute gradient components using static_cast for type safety and readability
    gx = static_cast<float>(GPU_i[idx - 1] - GPU_i[idx + 1]);
    gy = static_cast<float>(GPU_i[idx_prev] - GPU_i[idx_next]);

    // Calculate gradient magnitude
    Gradient[idx] = sqrtf(gx * gx + gy * gy);
    // Calculate gradient angle
    ang = atan2f(gy, gx);

    // Calculate display angle using the ternary operator for simplified control flow
    displayang = (ang < 0) ? 8 * (ang + PI) : 8 * ang;

    // Assign display orientation based on the display angle
    if (displayang < PI || displayang > 7 * PI)
      DisplayOrientation[idx] = 0;
    else if (displayang >= PI && displayang < 3 * PI)
      DisplayOrientation[idx] = 1;
    else if (displayang >= 3 * PI && displayang < 5 * PI)
      DisplayOrientation[idx] = 2;
    else
      DisplayOrientation[idx] = 3;

    // Adjust the angle if it's negative
    if (ang < 0){
      if (hp.Orientation == 0){
        ang = ang + PI;
      }
      else{
        ang = 2 * PI + ang;
      }
    }

    // Calculate bin index for the current angle based on the selected orientation mode
    if (hp.Orientation == 0)
      ang = (hp.NumBins)*ang / PI;
    else
      ang = (hp.NumBins)*ang / (2 * PI);
    
    // Store the bin index in the Orientation array
    Orientation[idx] = static_cast<int>(ang);
  }

}

//-------------------------------------------------------------Cell_kernel-------------------------------------------------------------------------

// Cell_kernel Original Version 0
__global__ void Cell_kernel_v0(float *histogram, int *Orientation, float *Gradient, HogProp hp){
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  int idx = i*hp.ImgCol*hp.CellSize + j*hp.CellSize;
  int idcell = i*hp.CellCol*hp.NumBins + j*hp.NumBins;
  int current_i,m,n;
  //int idx_next= (i+1)*hp.ImgCol + j;

  if(i<hp.CellRow & j<hp.CellCol) {
    for (m=0;m<hp.CellSize;m++) {
      current_i=idx+m*hp.ImgCol;
      for (n=0;n<hp.CellSize;n++) {
        histogram[idcell+Orientation[current_i+n]]+=Gradient[current_i+n];
      }
    }
  }
}

// Cell_kernel Optimized Version 1
// Significantly improved the execution time performance compared to original version 0
// Modified the thread configuration by adding the z-dimension and setting threadsPerBlock.z equal to hp.CellSize * hp.CellSize
// Takes full advantage of the GPU's three-dimensional parallelism, distributing the workload more evenly across the threads and increasing the overall throughput
// In the original Cell_kernel, there were two nested loops, which contributed to a higher degree of sequential processing
// By leveraging the z-dimension of the threads and assigning each thread to a unique combination of (m, n) indices, we were able to eliminate the need for these nested loops
// The optimized kernel has a more coherent memory access pattern, as each thread accesses consecutive memory locations when reading from and writing to the Orientation and Gradient arrays
// This improvement in memory access pattern reduces memory latency and contributes to the overall performance improvement
__global__ void Cell_kernel_v1(float *histogram, int *Orientation,float *Gradient, HogProp hp){
  // Calculate row, column, and cell indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col of image
  int k = threadIdx.z; // index within the cell

  // Ensure the current thread is within image and cell boundaries
  if (i < hp.CellRow && j < hp.CellCol && k < hp.CellSize * hp.CellSize){
    // Calculate cell_i and cell_j, which represent the local row and column indices within the cell
    int cell_i = k / hp.CellSize;
    int cell_j = k % hp.CellSize;

    // Calculate the global row and column indices in the image corresponding to the current thread
    int img_i = i * hp.CellSize + cell_i;
    int img_j = j * hp.CellSize + cell_j;

    // Calculate the linear indices for the image and the cell histogram
    int img_idx = img_i * hp.ImgCol + img_j;
    int cell_idx = i * hp.CellCol * hp.NumBins + j * hp.NumBins;

    // Update the cell histogram by accumulating gradient values based on their orientation
    histogram[cell_idx + Orientation[img_idx]] += Gradient[img_idx];
  }
}

// Cell_kernel Optimized Version 2
// Significantly improved the execution time performance compared to original version 0, and slightly improved from version 1
// Adds shared memory utilization to store the portion of Orientation and Gradient arrays required by each thread
// Reduces the number of global memory accesses
// Also includes the optimization methods of Version 1 described below
// Modified the thread configuration by adding the z-dimension and setting threadsPerBlock.z equal to hp.CellSize * hp.CellSize
// Takes full advantage of the GPU's three-dimensional parallelism, distributing the workload more evenly across the threads and increasing the overall throughput
// In the original Cell_kernel, there were two nested loops, which contributed to a higher degree of sequential processing
// By leveraging the z-dimension of the threads and assigning each thread to a unique combination of (m, n) indices, we were able to eliminate the need for these nested loops
// The optimized kernel has a more coherent memory access pattern, as each thread accesses consecutive memory locations when reading from and writing to the Orientation and Gradient arrays
// This improvement in memory access pattern reduces memory latency and contributes to the overall performance improvement
__global__ void Cell_kernel_v2(float *histogram, int *Orientation,float *Gradient, HogProp hp){
  // Calculate row, column, and cell indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col of image
  int k = threadIdx.z; // index within the cell

  // Allocate shared memory for Orientation and Gradient data
  __shared__ int shared_Orientation[BOX_SIZE][BOX_SIZE];
  __shared__ float shared_Gradient[BOX_SIZE][BOX_SIZE];

  // Ensure the current thread is within image and cell boundaries
  if (i < hp.CellRow && j < hp.CellCol && k < hp.CellSize * hp.CellSize) {
    // Calculate cell_i and cell_j, which represent the local row and column indices within the cell
    int cell_i = k / hp.CellSize;
    int cell_j = k % hp.CellSize;

    // Calculate the global row and column indices in the image corresponding to the current thread
    int img_i = i * hp.CellSize + cell_i;
    int img_j = j * hp.CellSize + cell_j;

    // Calculate the linear indices for the image and the cell histogram
    int img_idx = img_i * hp.ImgCol + img_j;
    int cell_idx = i * hp.CellCol * hp.NumBins + j * hp.NumBins;

    // Load data from global memory to shared memory
    shared_Orientation[cell_i][cell_j] = Orientation[img_idx];
    shared_Gradient[cell_i][cell_j] = Gradient[img_idx];

    // Ensure that all data is loaded into shared memory before proceeding
    __syncthreads();

    // Perform the histogram computation using shared memory arrays
    histogram[cell_idx + shared_Orientation[cell_i][cell_j]] += shared_Gradient[cell_i][cell_j];
  }
}

//-------------------------------------------------------------Block_kernel-------------------------------------------------------------------------

// Block_kernel Original Version 0
__global__ void Block_kernel_v0(float *FinalFeatures, float *histogram, HogProp hp){

 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  
  int step=hp.BlockSize-hp.BlockOverlap;
  int idblock = i*hp.BlockCol*hp.FeatureSize + j*hp.FeatureSize;
  int idcell = i*hp.CellCol*step*hp.NumBins + j*step*hp.NumBins;
  int current_i,current_j,m,n;
  float average=0.000000001;
  int horz=hp.BlockSize*hp.NumBins;
  //int idx_next= (i+1)*hp.ImgCol + j;
  
  if(i<hp.BlockRow & j<hp.BlockCol) {
    for (m=0;m<hp.BlockSize;m++) {
      current_i=idcell+m*hp.CellCol*hp.NumBins;
      for (n=0;n<horz;n++) {
        average=average+histogram[current_i+n];
      }
    }
  }
  
  if(i<hp.BlockRow & j<hp.BlockCol) {
    for (m=0;m<hp.BlockSize;m++) {
      current_i=idcell+m*hp.CellCol*hp.NumBins;
      current_j=idblock+m*hp.CellCol;
      for (n=0;n<horz;n++) {
        FinalFeatures[current_j+n]=histogram[current_i+n]/average;
      }
    }
  }
}

// Block_kernel Optimized Version 1
// Seemed to increase the execution time slightly giving worse performance than Version 0
// Reduction of redundant memory accesses by utilizing shared memory to store histogram values
// Merged the two separate loops into a single loop to streamline the code
__global__ void Block_kernel_v1(float *FinalFeatures, float *histogram, HogProp hp){
  // Calculate row and column indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x; // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y; // col of image

  // Calculate the step size, idblock, and idcell for the current thread
  int step = hp.BlockSize - hp.BlockOverlap;
  int idblock = i * hp.BlockCol * hp.FeatureSize + j * hp.FeatureSize;
  int idcell = i * hp.CellCol * step * hp.NumBins + j * step * hp.NumBins;

  // Initialize loop variables and the average value
  int current_i, current_j, m, n;
  float average = 0.000000001f;
  int horz = hp.BlockSize * hp.NumBins;

  // Allocate shared memory for histogram values
  __shared__ float shHistogram[256]; // assuming the maximum value of hp.BlockSize * hp.NumBins is 256
  int threadId = threadIdx.x * blockDim.y + threadIdx.y;

  // Ensure the current thread is within the block row and column boundaries
  if (i < hp.BlockRow && j < hp.BlockCol){
    // Combine the previously separate loops into a single loop for streamlined code
    for (m = 0; m < hp.BlockSize; m++){
      // Calculate the current_i and current_j values for the loop iteration
      current_i = idcell + m * hp.CellCol * hp.NumBins;
      current_j = idblock + m * hp.CellCol;

      // Load histogram values into shared memory and compute the final features
      for (n = 0; n < horz; n++){
        // Load histogram values into shared memory if threadId is within the valid range
        if (threadId < 256){
          shHistogram[threadId] = histogram[current_i + n];
        }
        __syncthreads(); // Ensure all threads have loaded the data before proceeding

        // Update the average value using the shared memory histogram values
        average += shHistogram[threadId];
        __syncthreads(); // Ensure all threads have updated the average value before proceeding

        // Compute the final feature value and store it in FinalFeatures
        FinalFeatures[current_j + n] = shHistogram[threadId] / average;
      }
    }
  }
}

// Block_kernel Optimized Version 2
// Improved execution time compared to original version 0 and 1
// Using global memory to optimize the kernel by reducing the number of calculations inside the loop
// Calculates the inverse of the average value outside the loop and then multiplies it with the histogram values to normalize them
__global__ void Block_kernel_v2(float *FinalFeatures, float *histogram, HogProp hp){
  // Calculate row and column indices for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image

  // Calculate the step size, idblock, and idcell for the current thread
  int step = hp.BlockSize - hp.BlockOverlap;
  int idblock = i * hp.BlockCol * hp.FeatureSize + j * hp.FeatureSize;
  int idcell = i * hp.CellCol * step * hp.NumBins + j * step * hp.NumBins;

  // Initialize loop variables and the average value
  int current_i, current_j, m, n;
  float average = 0.000000001;
  int horz = hp.BlockSize * hp.NumBins;

  // Ensure the current thread is within the block row and column boundaries
  if (i < hp.BlockRow && j < hp.BlockCol) {
    // Compute the average value using the histogram values from global memory
    for (m = 0; m < hp.BlockSize; m++) {
      current_i = idcell + m * hp.CellCol * hp.NumBins;
      for (n = 0; n < horz; n++) {
        average = average + histogram[current_i + n];
      }
    }

    // Calculate the inverse of the average value
    float inv_average = 1.0f / average;

    // Normalize the histogram values using the inverse of the average value and store the result in FinalFeatures
    for (m = 0; m < hp.BlockSize; m++) {
      current_i = idcell + m * hp.CellCol * hp.NumBins;
      current_j = idblock + m * hp.CellCol;
      for (n = 0; n < horz; n++) {
        FinalFeatures[current_j + n] = histogram[current_i + n] * inv_average;
      }
    }
  }
}

// Block_kernel Optimized Version 3
// Applies the Subhistogram accumulation method to the Block_kernel
// Reduced global memory accesses
// Each thread is assigned to a unique index in the final histogram, there are no conflicts or need for atomic operations
// No synchronization needed, there are no shared memory dependencies or conflicts that would necessitate synchronization among threads
// The optimized Version 3 Block_kernel has a simpler code structure
// Improved performance significantly vs. Version 0 and also improved vs. Version 1 and 2
__global__ void Block_kernel_v3(float *FinalFeatures, float *histogram, HogProp hp){
  // Calculate the block and thread indices for the current thread
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Calculate the unique blockId, cellIdX, and cellIdY for the current thread
  int blockId = by * hp.BlockCol + bx;
  int cellIdX = bx * (hp.BlockSize - hp.BlockOverlap) + tx;
  int cellIdY = by * (hp.BlockSize - hp.BlockOverlap) + ty;

  // Ensure the current thread is within the cell row and column boundaries
  if (cellIdX < hp.CellCol && cellIdY < hp.CellRow){
    // Calculate the unique cellId and histIndex for the current thread
    int cellId = cellIdY * hp.CellCol + cellIdX;
    int histIndex = blockId * hp.FeatureSize + ty * hp.CellCol * hp.NumBins + tx * hp.NumBins;

    // Accumulate the subhistograms into the final histogram
    for (int i = 0; i < hp.NumBins; i++){
      FinalFeatures[histIndex + i] += histogram[cellId * hp.NumBins + i];
    }
  }
}

//-------------------------------------------------------------Display_Cell_kernel-------------------------------------------------------------------------

// Display_Cell_kernel Original Version 0
__global__ void Display_Cell_kernel_v0(float* Displayhistogram, float *TempDisplayhistogram, uchar *DisplayOrientation,float *Gradient, DisplayProp dp){
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
   
  int idx = i*dp.ImgCol + j*dp.CellSize;
  int idxtemp = i*dp.CellCol*dp.NumBins*dp.CellSize + j*dp.NumBins;
  int idcell = i*dp.CellCol*dp.NumBins + j*dp.NumBins;
  int n;
  int temp_rowsize=dp.CellCol*dp.NumBins;
  //float avg;
  float max1,max2,avg;
  //int idx_next= (i+1)*hp.ImgCol + j;
  
  if(i<dp.HorzCells & j<dp.CellCol) {
    TempDisplayhistogram[idcell]=0; TempDisplayhistogram[idcell+1]=0; TempDisplayhistogram[idcell+2]=0; TempDisplayhistogram[idcell+3]=0;
    for (n=0;n<dp.CellSize;n++) {
      TempDisplayhistogram[idcell+DisplayOrientation[idx+n]]+=Gradient[idx+n];
    }
  }
  
  __syncthreads();
  
  if(i<dp.CellRow) {
    for(n=0;n<dp.CellSize;n++) {
      Displayhistogram[idcell]+=TempDisplayhistogram[idxtemp+n*temp_rowsize];
      Displayhistogram[idcell+1]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+1];
      Displayhistogram[idcell+2]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+2];
      Displayhistogram[idcell+3]+=TempDisplayhistogram[idxtemp+n*temp_rowsize+3];
    }
    
    if(Displayhistogram[idcell]>Displayhistogram[idcell+1]) {max1=Displayhistogram[idcell];}   else {max1=Displayhistogram[idcell+1];}
    if(Displayhistogram[idcell+2]>Displayhistogram[idcell+3]) {max2=Displayhistogram[idcell+2];} else {max2=Displayhistogram[idcell+3];}
    if(max2>max1) max1=max2;
    avg=max1/8;
    //avg=(Displayhistogram[idcell+3]+Displayhistogram[idcell+2]+Displayhistogram[idcell+1]+Displayhistogram[idcell])/8;
    //avg=1;
    if(Displayhistogram[idcell+3]>=0) Displayhistogram[idcell+3]=Displayhistogram[idcell+3]/avg; else Displayhistogram[idcell+3]=0;
    if(Displayhistogram[idcell+2]>=0) Displayhistogram[idcell+2]=Displayhistogram[idcell+2]/avg; else Displayhistogram[idcell+2]=0;
    if(Displayhistogram[idcell+1]>=0) Displayhistogram[idcell+1]=Displayhistogram[idcell+1]/avg; else Displayhistogram[idcell+1]=0;
    if(Displayhistogram[idcell]>=0) Displayhistogram[idcell]=Displayhistogram[idcell]/avg; else Displayhistogram[idcell]=0;
  }
}

// Display_Cell_kernel Optimized Version 1
// No changes in performance execution time
// Reducing redundant calculations and removing unnecessary branches
// Combined the initialization of TempDisplayhistogram elements with the first loop, removing the need for a separate loop
// Replaced the branches with the use of fmaxf function to find the maximum values and updated the normalization step to use a single loop
__global__ void Display_Cell_kernel_v1(float* Displayhistogram, float *TempDisplayhistogram, uchar *DisplayOrientation,float *Gradient, DisplayProp dp){
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image

  int idx = i * dp.ImgCol + j * dp.CellSize;
  int idxtemp = i * dp.CellCol * dp.NumBins * dp.CellSize + j * dp.NumBins;
  int idcell = i * dp.CellCol * dp.NumBins + j * dp.NumBins;
  int n;
  int temp_rowsize = dp.CellCol * dp.NumBins;
  float max1, max2, avg;

  // Initialize TempDisplayhistogram elements and calculate histogram
  if (i < dp.HorzCells && j < dp.CellCol) {
    for (int k = 0; k < dp.NumBins; k++) {
      TempDisplayhistogram[idcell + k] = 0;
    }

    for (n = 0; n < dp.CellSize; n++) {
      TempDisplayhistogram[idcell + DisplayOrientation[idx + n]] += Gradient[idx + n];
    }
  }

  __syncthreads();

  // Combine histograms and normalize
  if (i < dp.CellRow) {
    for (n = 0; n < dp.CellSize; n++) {
      for (int k = 0; k < dp.NumBins; k++) {
        Displayhistogram[idcell + k] += TempDisplayhistogram[idxtemp + n * temp_rowsize + k];
      }
    }

    // Calculate the maximum value using fmaxf function
    max1 = fmaxf(Displayhistogram[idcell], Displayhistogram[idcell + 1]);
    max2 = fmaxf(Displayhistogram[idcell + 2], Displayhistogram[idcell + 3]);
    avg = fmaxf(max1, max2) / 8;

    // Normalize the histogram values
    for (int k = 0; k < dp.NumBins; k++) {
      Displayhistogram[idcell + k] = fmaxf(Displayhistogram[idcell + k] / avg, 0);
    }
  }
}

//-------------------------------------------------------------display_kernel-------------------------------------------------------------------------

// display_kernel Original Version 0
__global__ void display_kernel_v0(float *Displayhistogram, uchar *GPU_odata, DisplayProp dp){
 	int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
	int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  int k = threadIdx.z;
   
  int idx = i*dp.CellCol*4 + j*4+k;
  int idcell = i*dp.DisplayCellSize*dp.DisplayImgCol + j*dp.DisplayCellSize;
  int m;
  int temp=(int)Displayhistogram[idx];
  int tempid;
  
  tempid=idcell+8+8*dp.DisplayImgCol;
  for(m=1;m<temp ;m++) {
    if(k==0) {
      GPU_odata[tempid+m]=255; GPU_odata[tempid-m]=255;
    }else if(k==1) {
      GPU_odata[tempid+m-m*dp.DisplayImgCol]=255; GPU_odata[tempid-m+m*dp.DisplayImgCol]=255;
    }else if(k==2) {
      GPU_odata[tempid-m*dp.DisplayImgCol]=255; GPU_odata[tempid+m*dp.DisplayImgCol]=255;
    }else {
      GPU_odata[tempid+m+m*dp.DisplayImgCol]=255; GPU_odata[tempid+m+m*dp.DisplayImgCol]=255;
    }
  }
  if(k==0) GPU_odata[tempid]=255;
}

// display_kernel Optimized Version 1
// Instead of having a separate if block for each value of k, 
// A single block is used that computes the offset based on the value of k
// This will reduce the number of branches in the code, which may help improve performance
// Did not provide any performance changes
__global__ void display_kernel_v1(float *Displayhistogram, uchar *GPU_odata, DisplayProp dp){
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  int k = threadIdx.z;
  
  int idx = i * dp.CellCol * 4 + j * 4 + k;
  int idcell = i * dp.DisplayCellSize * dp.DisplayImgCol + j * dp.DisplayCellSize;
  int m;
  int temp = (int)Displayhistogram[idx];
  int tempid;

  tempid = idcell + 8 + 8 * dp.DisplayImgCol;

  // Define offset arrays for x and y based on the value of k
  int offset_x[] = {1, 1, 0, -1};
  int offset_y[] = {0, 1, 1, 1};

  // Draw lines for each value of k using the offset arrays
  for(m = 1; m < temp; m++) {
    int x1 = tempid + m * offset_x[k];
    int x2 = tempid - m * offset_x[k];
    int y1 = m * offset_y[k];
    int y2 = -m * offset_y[k];

    GPU_odata[x1 + y1 * dp.DisplayImgCol] = 255;
    GPU_odata[x2 + y2 * dp.DisplayImgCol] = 255;
  }
  
  // Set the center pixel to 255 for k == 0
  if(k == 0) GPU_odata[tempid] = 255;
}

// display_kernel Optimized Version 2
// Moved the calculations of the offsets outside of the loop and used accumulative sums to reduce the number of calculations inside the loop
// Did not provide any performance changes
__global__ void display_kernel_v2(float *Displayhistogram, uchar *GPU_odata, DisplayProp dp){
  // Calculate the row, column, and bin index for the current thread
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
  int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image
  int k = threadIdx.z;  // bin index

  // Calculate the histogram index and display cell index for the current thread
  int idx = i * dp.CellCol * 4 + j * 4 + k;
  int idcell = i * dp.DisplayCellSize * dp.DisplayImgCol + j * dp.DisplayCellSize;

  int m;  // loop variable for iterating through the histogram values
  int temp = (int)Displayhistogram[idx];  // Get the histogram value at the current index
  int tempid;  // temporary variable to store the pixel index in the display image

  // Calculate the starting index in the display image
  tempid = idcell + 8 + 8 * dp.DisplayImgCol;

  // Define the x and y offset arrays based on the bin index (k)
  int offset_x[] = {1, 1, 0, -1};
  int offset_y[] = {0, 1, 1, 1};

  // Initialize base values for x and y coordinates
  int base_x1 = tempid;
  int base_x2 = tempid;
  int base_y1 = 0;
  int base_y2 = 0;

  // Iterate through the values of m and use accumulative sums for x and y coordinates
  for(m = 1; m < temp; m++) {
    // Update the x and y coordinates using the accumulative sums
    base_x1 += offset_x[k];
    base_x2 -= offset_x[k];
    base_y1 += offset_y[k];
    base_y2 -= offset_y[k];

    // Set the corresponding pixels in the display image to 255 (white) based on the updated x and y coordinates
    GPU_odata[base_x1 + base_y1 * dp.DisplayImgCol] = 255;
    GPU_odata[base_x2 + base_y2 * dp.DisplayImgCol] = 255;
  }
  
  // Set the center pixel to 255 (white) for k == 0
  if(k == 0) GPU_odata[tempid] = 255;
}















// hogFeature takes in Mat image and returns the Mat image of the HOG features extracted
Mat hogFeature(Mat image, std::string filename){
	
	//-------------------------------------------------------------variables-------------------------------------------------------------------------
	//int i;
  float GPURuntimes[7];
	//===============================================================================================================================================

  // No global stuff
  struct HogProp hp;
  struct DisplayProp dp;
  uchar * CPU_InputArray;//, * CPU_OutputArray;
  float *CPU_FeatureArray;//, *CPU_Hist;

  // Setting input parameters adjusted optimized performance set
  hp.CellSize= 8; //atoi(argv[3]); 
  hp.BlockSize= 2; //atoi(argv[4]); 
  hp.BlockOverlap= 1; //atoi(argv[5]); 		  
  hp.NumBins= 9; //atoi(argv[6]); 
  hp.Orientation= 0; //atoi(argv[7]);

  // Using optimized kernels versions for performance set
  Cal_kernel_v = 0; //atoi(argv[3]); 0 or 1 
  Cell_kernel_v = 0; //atoi(argv[4]);
  Block_kernel_v = 0; //atoi(argv[5]);
  //Display_Cell_kernel_v = 0; //atoi(argv[6]);
  //display_kernel_v = 0; //atoi(argv[7]);

	if(! image.data ) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}

	hp.ImgRow=image.rows;
	hp.ImgCol=image.cols;
  hp.ImgSize=hp.ImgRow*hp.ImgCol;
  hp.CellRow=floor(image.rows/hp.CellSize);
  hp.CellCol=floor(image.cols/hp.CellSize);
  hp.TotalCells=hp.CellRow*hp.CellCol;
	hp.BlockRow=(hp.CellRow-hp.BlockSize+1)/(hp.BlockSize-hp.BlockOverlap);
  hp.BlockCol=(hp.CellCol-hp.BlockSize+1)/(hp.BlockSize-hp.BlockOverlap);
  hp.TotalBlocks=hp.BlockRow*hp.BlockCol;
  hp.FeatureSize=hp.NumBins*hp.BlockSize*hp.BlockSize;

  // printf("-----------------------------------------------------------------------------------\n\n");

  //===============================================================================================================================================	

	//---------------------------------------------------Create CPU memory to store the output-------------------------------------------------------
	
  checkCudaErrors(cudaMallocHost ((void**)&CPU_InputArray,hp.ImgSize));
  //checkCudaErrors(cudaMallocHost ((void**)&CPU_OutputArray,dp.DisplayImgSize));	
  //checkCudaErrors(cudaMallocHost ((void**)&CPU_Hist,dp.TotalCells *4*4));	
  checkCudaErrors(cudaMallocHost ((void**)&CPU_FeatureArray,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));	
  memcpy(CPU_InputArray,image.data,hp.ImgSize);
  checkCudaErrors(launch_helper(GPURuntimes, CPU_FeatureArray, CPU_InputArray, dp, hp));


  printf("-----------------------------------------------------------------\n");
  printf("Cal= %d ... Cell= %d ... Block= %d \n",Cal_kernel_v, Cell_kernel_v, Block_kernel_v);
  printf("Tfr CPU->GPU = %5.2f ms...Execution = %5.2f ms...Tfr GPU->CPU = %5.2f ms...Total=%5.2f ms\n",
          GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
  printf("Avg-Cal=%5.4f ms...Avg-Cell=%5.4f ms...Avg-Block=%5.4f ms",
                      GPURuntimes[4], GPURuntimes[5],GPURuntimes[6]);
  printf("-----------------------------------------------------------------\n");

  //checkCudaErrors(launch_helper(GPURuntimes, CPU_FeatureArray, CPU_Hist, CPU_InputArray, CPU_OutputArray, dp, hp));

  // Output the HOG features to the SVM classifier 
  //Mat mat1d = train_samples[i].reshape(0, 1);
  Mat hogFeatureOutput = Mat(hp.TotalBlocks, hp.FeatureSize , CV_32F, CPU_FeatureArray).reshape(0, 1);
  // cv::FileStorage fs(filename, cv::FileStorage::WRITE);
  // fs<< "data" << hogFeatureOutput;
  // fs.release();
  //cudaFreeHost(CPU_OutputArray);
  cudaFreeHost(CPU_InputArray);
  //cudaFreeHost(CPU_Hist);	
	cudaFreeHost(CPU_FeatureArray);

  return hogFeatureOutput; // return the HOG Feature image output for the SVM
}

cudaError_t launch_helper(float* Runtimes, float *CPU_FeatureArray, uchar *CPU_InputArray, DisplayProp dp, HogProp hp){
	cudaEvent_t time1, time2, time3, time4, time5, time6, time7;

  int   *Orientation;
	float *Gradient;
  uchar *DisplayOrientation;
	uchar *GPU_idata;
	//uchar *GPU_odata;
 	//uchar *GPU_displaydata;
  float *GPU_CellHistogram;
  float *GPU_BlockHistogram;
  // float *TempDisplayhistogram;
  // float *Displayhistogram;
  dim3 threadsPerBlock;
	dim3 numBlocks;
  int i;
  //cudaStream_t stream[2];

  cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);  
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
 
  cudaEventCreate(&time1);
  cudaEventCreate(&time2);
  cudaEventCreate(&time3);
  cudaEventCreate(&time4);
  cudaEventCreate(&time5);
  cudaEventCreate(&time6);
  cudaEventCreate(&time7);
  
 
 cudaEventRecord(time1, 0);
 
 //for(i=0;i<2;i++) checkCudaErrors(cudaStreamCreate(&stream[i]));
 
 checkCudaErrors(cudaMalloc((void**)&GPU_idata, hp.ImgSize));
 checkCudaErrors(cudaMalloc((void**)&Gradient, hp.ImgSize*4));
 checkCudaErrors(cudaMalloc((void**)&Orientation, hp.ImgSize*4));
 checkCudaErrors(cudaMalloc((void**)&DisplayOrientation, hp.ImgSize));
// checkCudaErrors(cudaMemcpyAsync(GPU_idata, CPU_InputArray, hp.ImgSize, cudaMemcpyHostToDevice,stream[0]));
checkCudaErrors(cudaMemcpyAsync(GPU_idata, CPU_InputArray, hp.ImgSize, cudaMemcpyHostToDevice));
 cudaEventRecord(time2, 0);

 //-------------------------------------------------------------Cal_kernel-------------------------------------------------------------------------
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.ImgRow / (float)threadsPerBlock.x), (int)ceil(hp.ImgCol / (float)threadsPerBlock.y));

 cudaEventRecord(time5, 0);
 if(Cal_kernel_v==1){
  //Cal_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
    Cal_kernel_v1<<<numBlocks, threadsPerBlock,0>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
 } else {
  //Cal_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
  Cal_kernel_v0<<<numBlocks, threadsPerBlock,0>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
 }

 checkCudaErrors(cudaDeviceSynchronize());

 cudaFree(GPU_idata);
 cudaEventRecord(time6, 0);
  //-------------------------------------------------------------Cell_kernel-------------------------------------------------------------------------
 // Cell_kernel
 checkCudaErrors(cudaMallocHost ((void**)&GPU_CellHistogram,hp.TotalCells*sizeof(float)*hp.NumBins));

 // Call the kernel
 if(Cell_kernel_v==1){
  // Optimized Cell_kernel 3D
  // Update threadsPerBlock to include the hp.CellSize in the z-dimension
  threadsPerBlock.x = BOX_SIZE;
  threadsPerBlock.y = BOX_SIZE;
  threadsPerBlock.z = hp.CellSize * hp.CellSize;

  // Update numBlocks definition
  numBlocks.x = (int)ceil(hp.CellRow / (float)threadsPerBlock.x);
  numBlocks.y = (int)ceil(hp.CellCol / (float)threadsPerBlock.y);

  //Cell_kernel_v1<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
  Cell_kernel_v1<<<numBlocks, threadsPerBlock, 0>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 } else if(Cell_kernel_v==2){
  // Optimized Cell_kernel 3D
  // Update threadsPerBlock to include the hp.CellSize in the z-dimension
  threadsPerBlock.x = BOX_SIZE;
  threadsPerBlock.y = BOX_SIZE;
  threadsPerBlock.z = hp.CellSize * hp.CellSize;

  // Update numBlocks definition
  numBlocks.x = (int)ceil(hp.CellRow / (float)threadsPerBlock.x);
  numBlocks.y = (int)ceil(hp.CellCol / (float)threadsPerBlock.y);

  //Cell_kernel_v2<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
    Cell_kernel_v2<<<numBlocks, threadsPerBlock, 0>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 } else {
  threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
  numBlocks = dim3((int)ceil(hp.CellRow / (float)threadsPerBlock.x), (int)ceil(hp.CellCol / (float)threadsPerBlock.y));
  //Cell_kernel_v0<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
    Cell_kernel_v0<<<numBlocks, threadsPerBlock, 0>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 }
 cudaEventRecord(time7, 0);
                                                                                                                                                        //

 //-------------------------------------------------------------Block_kernel-------------------------------------------------------------------------
 // Block_kernel
 checkCudaErrors(cudaMallocHost ((void**)&GPU_BlockHistogram,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.BlockRow / (float)threadsPerBlock.x), (int)ceil(hp.BlockCol / (float)threadsPerBlock.y));
 //print("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 
 //print("\n\n...%d %d...\n\n",hp.BlockRow,hp.BlockCol); 

 if(Block_kernel_v==1){
  //Block_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
    Block_kernel_v1<<<numBlocks, threadsPerBlock,0>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else if(Block_kernel_v==2){
  //Block_kernel_v2<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
    Block_kernel_v2<<<numBlocks, threadsPerBlock,0>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else if(Block_kernel_v==3){
 // Block_kernel_v3<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
     Block_kernel_v3<<<numBlocks, threadsPerBlock,0>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else {
//  Block_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
    Block_kernel_v0<<<numBlocks, threadsPerBlock,0>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 }

 //-------------------------------------------------------------Timings-------------------------------------------------------------------------
 cudaEventRecord(time3, 0);
 
//  checkCudaErrors(cudaMemcpyAsync(CPU_OutputArray, GPU_odata, dp.DisplayImgSize, cudaMemcpyDeviceToHost,stream[1]));
 checkCudaErrors(cudaDeviceSynchronize());

 checkCudaErrors(cudaMemcpy(CPU_FeatureArray,GPU_BlockHistogram , hp.TotalBlocks*sizeof(float)*hp.FeatureSize, cudaMemcpyDeviceToHost));
 checkCudaErrors(cudaDeviceSynchronize());
 	cudaEventRecord(time4, 0);
	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time5);
	cudaEventSynchronize(time6);
  cudaEventSynchronize(time7);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;
  float calTime, cellTime, blockTime;

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);
  cudaEventElapsedTime(&calTime, time5, time6);
  cudaEventElapsedTime(&cellTime, time6, time7);
  cudaEventElapsedTime(&blockTime, time7, time3);
  

	Runtimes[0] = totalTime;
	Runtimes[1] = tfrCPUtoGPU;
	Runtimes[2] = kernelExecutionTime;
	Runtimes[3] = tfrGPUtoCPU;
  Runtimes[4] = calTime;
  Runtimes[5] = cellTime;
  Runtimes[6] = blockTime;


 	Error:
//  for(i=0;i<2;i++) cudaStreamDestroy(stream[i]);
// 	cudaFree(GPU_odata);
// 	cudaFree(GPU_idata);
  cudaFree(Orientation);
  cudaFree(Gradient);
  cudaFree(DisplayOrientation);
  // cudaFree(GPU_BlockHistogram);
// 	cudaFree(Displayhistogram);
//   cudaFree(TempDisplayhistogram);
 	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
 
  return cudaStatus;
}