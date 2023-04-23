#include <stdio.h>
#include <string.h>

// CUDA stuff:
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenCV stuff (note: C++ not C):
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//other support
#include "HogSupport.h"
#define BOX_SIZE 8
cudaError_t launch_helper(float* Runtimes);

struct HogProp hp;
struct DisplayProp dp;
uchar * CPU_InputArray, * CPU_OutputArray;
float *CPU_CellArray,*CPU_FeatureArray, *CPU_Hist;

cudaStream_t stream[2];

using namespace cv;

int Cal_kernel_v;
int Cell_kernel_v;
int Block_kernel_v;
int Display_Cell_kernel_v;
int display_kernel_v;

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
__global__ void Cell_kernel_v0(float *histogram, int *Orientation,float *Gradient, HogProp hp){
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

// Display_Cell_kernel Optimized Version 2
// Attempted to optimize the kernel by utilizing shared memory to store the temporary histogram values, which may help reduce global memory access
// CUDA Runtime Error: an illegal memory access was encountered
// ...: cudaError_t checkCuda(cudaError_t): Assertion `result == cudaSuccess' failed.
// Aborted
// Used Version 1 instead
// __global__ void Display_Cell_kernel_v2(float* Displayhistogram, float *TempDisplayhistogram, uchar *DisplayOrientation,float *Gradient, DisplayProp dp){
//   int i = blockIdx.x * blockDim.x + threadIdx.x;  // row of image
//   int j = blockIdx.y * blockDim.y + threadIdx.y;  // col of image

//   int idx = i * dp.ImgCol + j * dp.CellSize;
//   int idcell = i * dp.CellCol * dp.NumBins + j * dp.NumBins;
//   int n;
//   float max1, max2, avg;

//   // Allocate shared memory
//   extern __shared__ float shared_histogram[];

//   // Initialize shared_histogram and calculate histogram
//   // Be cautious about indexing shared_histogram; make sure it doesn't access memory beyond its allocated bounds
//   if (i < dp.HorzCells && j < dp.CellCol) {
//     for (int k = 0; k < dp.NumBins; k++) {
//       shared_histogram[threadIdx.x * dp.CellCol * dp.NumBins + threadIdx.y * dp.NumBins + k] = 0;
//     }

//     for (n = 0; n < dp.CellSize; n++) {
//       shared_histogram[threadIdx.x * dp.CellCol * dp.NumBins + threadIdx.y * dp.NumBins + DisplayOrientation[idx + n]] += Gradient[idx + n];
//     }
//   }

//   __syncthreads();

//   // Update Displayhistogram from shared_histogram
//   if (i < dp.CellRow) {
//     for (n = 0; n < dp.CellSize; n++) {
//       for (int k = 0; k < dp.NumBins; k++) {
//         Displayhistogram[idcell + k] += shared_histogram[threadIdx.x * dp.CellCol * dp.NumBins + threadIdx.y * dp.NumBins + k];
//       }
//     }

//     max1 = fmaxf(Displayhistogram[idcell], Displayhistogram[idcell + 1]);
//     max2 = fmaxf(Displayhistogram[idcell + 2], Displayhistogram[idcell + 3]);
//     avg = fmaxf(max1, max2) / 8;

//     for (int k = 0; k < dp.NumBins; k++) {
//       Displayhistogram[idcell + k] = fmaxf(Displayhistogram[idcell + k] / avg, 0);
//     }
//   }
// }

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
Mat hogFeature(char *argv[]){
  //----------------------------------------------------------------Load Image---------------------------------------------------------------------
  Mat image;	// see http://docs.opencv.org/modules/core/doc/basic_structures.html#mat
  const char* imageFile = "test.jpg";
  image = imread(imageFile, CV_LOAD_IMAGE_GRAYSCALE); //Load Grayscale image argv[1]
    
	//-------------------------------------------------------------variables-------------------------------------------------------------------------
	//int i;
  float GPURuntimes[4];
	//===============================================================================================================================================
	
	//--------------------------------------------------Input Parameter error chec-------------------------------------------------------------------
	// switch (argc){
	// 	case 3 : hp.CellSize=8; 			hp.BlockSize=2; 			hp.BlockOverlap=1; 					  hp.NumBins=9; 			hp.Orientation=0; 			  break;
	// 	case 4 : hp.CellSize=atoi(argv[3]); hp.BlockSize=2; 			hp.BlockOverlap=1; 			   		  hp.NumBins=9; 			hp.Orientation=0; 			  break;
	// 	case 5 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=ceil(hp.BlockSize/2); hp.NumBins=9; 			hp.Orientation=0; 			  break;
	// 	case 6 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=9; 			hp.Orientation=0; 			  break;
	// 	case 7 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=atoi(argv[6]); hp.Orientation=0; 			  break;
	// 	case 8 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=atoi(argv[6]); hp.Orientation=atoi(argv[7]); break;
  //   case 9 : hp.CellSize=atoi(argv[3]); hp.BlockSize=atoi(argv[4]); hp.BlockOverlap=atoi(argv[5]); 		  hp.NumBins=atoi(argv[6]); hp.Orientation=atoi(argv[7]); break;
	// 	default: printf("\n\nUsage: hogfeature <inputimage> <output image> <Cell Size> <Block Size> <Block Overlap> <Number of Bins> <Orintation>");
	// 	printf("\n\nExample: hogfeature infilename.bmp outname.bmp 8 2 1 9 0\n");
	// 	printf("\nNumber of input parameters must be between 2 and 7\n");
	// 	return 0;
	// }

  // Default example settings
  // hp.CellSize= 8; //atoi(argv[3]); 
  // hp.BlockSize= 2; //atoi(argv[4]); 
  // hp.BlockOverlap= 1; //atoi(argv[5]); 		  
  // hp.NumBins= 9; //atoi(argv[6]); 
  // hp.Orientation= 0; //atoi(argv[7]);

  // Using input arguments for collecting results comparison data for all of them
  Cal_kernel_v = atoi(argv[1]);
  Cell_kernel_v = atoi(argv[2]);
  Block_kernel_v = atoi(argv[3]);
  Display_Cell_kernel_v = atoi(argv[4]);
  display_kernel_v = atoi(argv[5]);

  // Setting input parameters adjusted optimized performance set
  hp.CellSize= 8; //atoi(argv[3]); 
  hp.BlockSize= 8; //atoi(argv[4]); 
  hp.BlockOverlap= 0; //atoi(argv[5]); 		  
  hp.NumBins= 9; //atoi(argv[6]); 
  hp.Orientation= 0; //atoi(argv[7]);

  // Using optimized kernels versions for performance set
  // Cal_kernel_v = 1; //atoi(argv[3]);
  // Cell_kernel_v = 2; //atoi(argv[4]);
  // Block_kernel_v = 3; //atoi(argv[5]);
  // Display_Cell_kernel_v = 1; //atoi(argv[6]);
  // display_kernel_v = 1; //atoi(argv[7]);

  // Display kernels versions used for testing and collecting results
  printf("\n-----------------------------------------------------------");
  printf("\nRunning Cal_kernel_v # = %d", Cal_kernel_v);
  printf("\nRunning Cell_kernel_v # = %d", Cell_kernel_v);
  printf("\nRunning Block_kernel_v # = %d", Block_kernel_v);
  printf("\nRunning Display_Cell_kernel_v # = %d", Display_Cell_kernel_v);
  printf("\nRunning display_kernel_v # = %d", display_kernel_v);
  printf("\n-----------------------------------------------------------\n\n");


  if(Cal_kernel_v>1 || Cal_kernel_v<0) {
		printf("\n\nCal_kernel_v = %d is invalid\n\n", Cal_kernel_v);
		exit(EXIT_FAILURE);
	}

    if(Cell_kernel_v>2 || Cell_kernel_v<0) {
		printf("\n\nCell_kernel_v = %d is invalid\n\n", Cell_kernel_v);
		exit(EXIT_FAILURE);
	}

  if(Block_kernel_v>3 || Block_kernel_v<0) {
		printf("\n\nBlock_kernel_v = %d is invalid\n\n", Block_kernel_v);
		exit(EXIT_FAILURE);
	}

  if(Display_Cell_kernel_v>1 || Display_Cell_kernel_v<0) {
		printf("\n\nDisplay_Cell_kernel_v = %d is invalid\n\n", Display_Cell_kernel_v);
		exit(EXIT_FAILURE);
	}

  if(display_kernel_v>2 || display_kernel_v<0) {
		printf("\n\ndisplay_kernel_v = %d is invalid\n\n", display_kernel_v);
		exit(EXIT_FAILURE);
	}
	
  // Removed EXIT_FAILURE checks below since they are hard coded in now
	// if(hp.CellSize<2 | hp.CellSize> 32) {
	// 	printf("\n\nCellSize = %d is invalid",hp.CellSize);
	// 	printf("\n Cell Size can be an integer between 2 and 32\n");
	// 	exit(EXIT_FAILURE);
	// }
	
	// if(hp.BlockSize<0 | hp.BlockSize> 16) {
	// 	printf("\n\nBlockSize = %d is invalid",hp.BlockSize);
	// 	printf("\n Block Size can be an integer between 1 and 8\n");
	// 	exit(EXIT_FAILURE);
	// }

	// if(hp.BlockOverlap<0 | hp.BlockOverlap> hp.BlockSize-1) {
	// 	printf("\n\nBlockSize = %d is invalid",hp.BlockOverlap);
	// 	printf("\n Block overlap can be an integer between 0 and %d\n",hp.BlockSize-1);
	// 	exit(EXIT_FAILURE);
	// }
	
	// if(hp.NumBins<4 | hp.NumBins> 180) {
	// 	printf("\n\nNumBins = %d is invalid",hp.NumBins);
	// 	printf("\nNumber of bins can be an integer between 0 and 180\n");
	// 	exit(EXIT_FAILURE);
	// }
	
	// if(hp.Orientation<0 | hp.Orientation>1) {
	// 	printf("\n\nOrientation = %d is invalid",hp.Orientation);
	// 	printf("\n Orientation can be either 0 or 1\n");
	// 	exit(EXIT_FAILURE);
	// }
 
  // Comment printf out for project
  // printf("\n\nPARAMETERS:\n\n");
  // printf("CellSize=%d, BlockSize=%d, BlockOverlap=%d, NumBins=%d, Orientation=%d\n",hp.CellSize,hp.BlockSize,hp.BlockOverlap,hp.NumBins,hp.Orientation);
	//===============================================================================================================================================

	if(! image.data ) {
		fprintf(stderr, "Could not open or find the image.\n");
		exit(EXIT_FAILURE);
	}

  // Comment printf out for project
	// printf("Loaded image: size = %dx%d (dims = %d).\n\n", image.rows, image.cols, image.dims); // argv[1]

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

  // Comment printf out for project
  // printf("----------------------------------IMAGE DIVIDED INTO CELL HISTOGRAM----------------\n");
  // printf("\nCell_rows = %d, Cell_columns = %d, Total_cells = %d\n",hp.CellRow,hp.CellCol,hp.TotalCells);
	// printf("\nBlock_rows = %d, Block_columns = %d, Total_blocks = %d\n",hp.BlockRow,hp.BlockCol,hp.TotalBlocks);
  // printf("\nfeaturesize=%d\n",hp.FeatureSize);
  // printf("-----------------------------------------------------------------------------------\n\n");
  
  dp.ImgRow=hp.ImgRow;
  dp.ImgCol=hp.ImgCol;
  dp.ImgSize=hp.ImgSize;
  dp.CellRow=32;
  dp.CellSize=dp.ImgRow/dp.CellRow;
  dp.CellCol=dp.ImgCol/dp.CellSize;
  dp.TotalCells=dp.CellRow*dp.CellCol;
  dp.NumBins=4;
  dp.HorzCellsTotal=dp.CellSize*dp.TotalCells;
  dp.HorzCells=dp.CellSize*dp.CellRow;
  
  dp.DisplayCellSize=17;
  dp.DisplayImgRow=dp.DisplayCellSize*dp.CellRow;
  dp.DisplayImgCol=dp.DisplayCellSize*dp.CellCol;
  dp.DisplayImgSize=dp.DisplayImgCol*dp.DisplayImgRow;

  // Comment printf out for project
  // printf("----------------------IMAGE DIVIDED INTO CELL HISTOGRAM FOR DISPLAYING-------------\n");
  // printf("\nCell_rows = %d, Cell_columns = %d, Total_cells=%d, Cell_size=%d, Horz_cells=%d\n",dp.CellRow,dp.CellCol,dp.TotalCells,dp.CellSize,dp.HorzCells);
  // printf("\nDisplay_rows = %d, Display_columns = %d, Total_pixels=%d, Cell_size=%d\n",dp.DisplayImgRow,dp.DisplayImgCol,dp.DisplayImgSize,dp.DisplayCellSize);
  // printf("-----------------------------------------------------------------------------------\n\n");

  //===============================================================================================================================================	

	//---------------------------------------------------Create CPU memory to store the output-------------------------------------------------------
	
  checkCuda(cudaMallocHost ((void**)&CPU_InputArray,hp.ImgSize));
  checkCuda(cudaMallocHost ((void**)&CPU_OutputArray,dp.DisplayImgSize));	
  checkCuda(cudaMallocHost ((void**)&CPU_Hist,dp.TotalCells *4*4));	
  checkCuda(cudaMallocHost ((void**)&CPU_FeatureArray,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));	
  memcpy(CPU_InputArray,image.data,hp.ImgSize);
  checkCuda(launch_helper(GPURuntimes));
  
  // Comment printf out for project (still shown for testing)
	printf("------------------------HOG Feature------------------------\n");
	printf("Tfr CPU->GPU = %5.2f ms ... \nExecution = %5.2f ms ... \nTfr GPU->CPU = %5.2f ms   \n Total=%5.2f ms\n",
			GPURuntimes[1], GPURuntimes[2], GPURuntimes[3], GPURuntimes[0]);
	printf("-----------------------------------------------------------\n");

  // Output the HOG features to the SVM classifier 
  Mat hogFeatureOutput = Mat(dp.DisplayImgRow, dp.DisplayImgCol, CV_8UC1, CPU_OutputArray);

  if (!imwrite("output.bmp", hogFeatureOutput)) { //argv[2]
		fprintf(stderr, "couldn't write output to disk!\n");
		cudaFreeHost(CPU_OutputArray);
    cudaFreeHost(CPU_InputArray);
	  cudaFreeHost(CPU_FeatureArray);
		exit(EXIT_FAILURE);
	}
 
  WriteNumbers("Feature.txt",CPU_FeatureArray,hp.BlockRow,hp.BlockCol,hp.FeatureSize);
  WriteNumbers("Display_feature.txt",CPU_Hist,dp.CellRow,dp.CellCol,4);

  printf("\nHOG Features written to Feature.txt");
  printf("\nHOG Features written to Display_feature.txt");
  printf("\nHOG Features written to output.bmp\n\n");
 
  cudaFreeHost(CPU_OutputArray);
  cudaFreeHost(CPU_InputArray);
  cudaFreeHost(CPU_Hist);	
	cudaFreeHost(CPU_FeatureArray);
  // printf("\nEXECUTION COMPLETED\n\n");
  exit(EXIT_SUCCESS);

  return hogFeatureOutput; // return the HOG Feature image output for the SVM
}



//###################################################################################################################################################
//------------------------------------------------------------main function--------------------------------------------------------------------------
//###################################################################################################################################################
// Remove main below when integrating into other main.cpp to call hogFeature() there to combine projects
int main(int argc, char *argv[]) {
  // Hog Feature extraction, input the modified image binary then use HOG feature output for SVM
  hogFeature(argv);
}

cudaError_t launch_helper(float* Runtimes){
	cudaEvent_t time1, time2, time3, time4;

  int   *Orientation;
	float *Gradient;
  uchar *DisplayOrientation;
	uchar *GPU_idata;
	uchar *GPU_odata;
 	//uchar *GPU_displaydata;
  float *GPU_CellHistogram;
  float *GPU_BlockHistogram;
  float *TempDisplayhistogram;
  float *Displayhistogram;
  dim3 threadsPerBlock;
	dim3 numBlocks;
  int i;
  
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
 
 cudaEventRecord(time1, 0);
 
 for(i=0;i<2;i++) checkCuda(cudaStreamCreate(&stream[i]));
 
 checkCuda(cudaMalloc((void**)&GPU_idata, hp.ImgSize));
 checkCuda(cudaMalloc((void**)&Gradient, hp.ImgSize*4));
 checkCuda(cudaMalloc((void**)&Orientation, hp.ImgSize*4));
 checkCuda(cudaMalloc((void**)&DisplayOrientation, hp.ImgSize));
 //checkCuda(cudaMalloc((void**)&GPU_displaydata, hp.ImgSize));
 //checkCuda(cudaMallocHost ((void**)&GPU_CellHistogram,hp.TotalCells*sizeof(float)*hp.NumBins));
 checkCuda(cudaMemcpyAsync(GPU_idata, CPU_InputArray, hp.ImgSize, cudaMemcpyHostToDevice,stream[0]));
 cudaEventRecord(time2, 0);

 //-------------------------------------------------------------Cal_kernel-------------------------------------------------------------------------
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.ImgRow / (float)threadsPerBlock.x), (int)ceil(hp.ImgCol / (float)threadsPerBlock.y));

 if(Cal_kernel_v==1){
  Cal_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
 } else {
  Cal_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_idata,Orientation,Gradient,DisplayOrientation,hp);
 }

 checkCuda(cudaDeviceSynchronize());
 cudaFree(GPU_idata);
 
 //-------------------------------------------------------------Display_Cell_kernel-------------------------------------------------------------------------
 checkCuda(cudaMalloc((void**)&TempDisplayhistogram, dp.HorzCellsTotal*4*4));
 checkCuda(cudaMalloc((void**)&Displayhistogram, dp.TotalCells *4*4)); 

 // The original Display_Cell_kernel
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(dp.HorzCells / (float)threadsPerBlock.x), (int)ceil(dp.CellCol / (float)threadsPerBlock.y));

 if(Display_Cell_kernel_v==1){
  Display_Cell_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,TempDisplayhistogram,DisplayOrientation,Gradient,dp);
 } else {
  Display_Cell_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,TempDisplayhistogram,DisplayOrientation,Gradient,dp);
 }

 // Display Kernel optimizing launch version (no changes)
//  threadsPerBlock = dim3(16, 16);
//  numBlocks = dim3((int)ceil(dp.HorzCells / (float)threadsPerBlock.x), (int)ceil(dp.CellCol / (float)threadsPerBlock.y));
//  Display_Cell_kernel<<<numBlocks, threadsPerBlock>>>(Displayhistogram, TempDisplayhistogram, DisplayOrientation, Gradient, dp);

 // Display Kernel Version 2 (failed)
 // size_t shared_mem_size;
 // shared_mem_size = dp.CellRow * dp.CellCol * dp.NumBins * sizeof(float);
 // Display_Cell_kernel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(Displayhistogram, TempDisplayhistogram, DisplayOrientation, Gradient, dp);
 
 //-------------------------------------------------------------Cell_kernel-------------------------------------------------------------------------
 // Cell_kernel
 checkCuda(cudaMallocHost ((void**)&GPU_CellHistogram,hp.TotalCells*sizeof(float)*hp.NumBins));

// Original Cell_kernel
//  threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
//  numBlocks = dim3((int)ceil(hp.CellRow / (float)threadsPerBlock.x), (int)ceil(hp.CellCol / (float)threadsPerBlock.y));
//  //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 
//  Cell_kernel<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_CellHistogram,Orientation,Gradient,hp);

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

  Cell_kernel_v1<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 } else if(Cell_kernel_v==2){
  // Optimized Cell_kernel 3D
  // Update threadsPerBlock to include the hp.CellSize in the z-dimension
  threadsPerBlock.x = BOX_SIZE;
  threadsPerBlock.y = BOX_SIZE;
  threadsPerBlock.z = hp.CellSize * hp.CellSize;

  // Update numBlocks definition
  numBlocks.x = (int)ceil(hp.CellRow / (float)threadsPerBlock.x);
  numBlocks.y = (int)ceil(hp.CellCol / (float)threadsPerBlock.y);

  Cell_kernel_v2<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 } else {
  threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
  numBlocks = dim3((int)ceil(hp.CellRow / (float)threadsPerBlock.x), (int)ceil(hp.CellCol / (float)threadsPerBlock.y));
  Cell_kernel_v0<<<numBlocks, threadsPerBlock, 0, stream[0]>>>(GPU_CellHistogram, Orientation, Gradient, hp);
 }

 //-------------------------------------------------------------display_kernel-------------------------------------------------------------------------
  // display_kernel
 checkCuda(cudaDeviceSynchronize());
 checkCuda(cudaMemcpy(CPU_Hist,Displayhistogram , dp.TotalCells *4*4, cudaMemcpyDeviceToHost));
 
 cudaFree(TempDisplayhistogram);
 cudaFree(Orientation); cudaFree(Gradient);
 checkCuda(cudaMalloc((void**)&GPU_odata, dp.DisplayImgSize));
 cudaMemset(GPU_odata, 0, dp.DisplayImgSize);
 threadsPerBlock = dim3(4, 4, 4);
 numBlocks = dim3((int)ceil(dp.CellRow / (float)threadsPerBlock.x), (int)ceil(dp.CellCol / (float)threadsPerBlock.y));
 //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 

 if(display_kernel_v==1){
  display_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,GPU_odata,dp);
 } else if(display_kernel_v==2){
  display_kernel_v2<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,GPU_odata,dp);
 } else {
  display_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[1]>>>(Displayhistogram,GPU_odata,dp);
 }
 
 //-------------------------------------------------------------Block_kernel-------------------------------------------------------------------------
 // Block_kernel
 checkCuda(cudaMallocHost ((void**)&GPU_BlockHistogram,hp.TotalBlocks*sizeof(float)*hp.FeatureSize));
 threadsPerBlock = dim3(BOX_SIZE, BOX_SIZE);
 numBlocks = dim3((int)ceil(hp.BlockRow / (float)threadsPerBlock.x), (int)ceil(hp.BlockCol / (float)threadsPerBlock.y));
 //printf("\n\n...%d %d...\n\n",numBlocks.x,numBlocks.y); 

 if(Block_kernel_v==1){
  Block_kernel_v1<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else if(Block_kernel_v==2){
  Block_kernel_v2<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else if(Block_kernel_v==3){
  Block_kernel_v3<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 } else {
  Block_kernel_v0<<<numBlocks, threadsPerBlock,0,stream[0]>>>(GPU_BlockHistogram, GPU_CellHistogram, hp);
 }

 //-------------------------------------------------------------Timings-------------------------------------------------------------------------
 cudaEventRecord(time3, 0);
 
 checkCuda(cudaMemcpyAsync(CPU_OutputArray, GPU_odata, dp.DisplayImgSize, cudaMemcpyDeviceToHost,stream[1]));
 checkCuda(cudaDeviceSynchronize());
 
 //checkCuda(cudaMemcpy(CPU_CellArray,GPU_CellHistogram , hp.TotalCells*sizeof(float)*hp.NumBins, cudaMemcpyDeviceToHost));
 
 checkCuda(cudaMemcpy(CPU_FeatureArray,GPU_BlockHistogram , hp.TotalBlocks*sizeof(float)*hp.FeatureSize, cudaMemcpyDeviceToHost));
 
 	cudaEventRecord(time4, 0);
	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	float totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime;

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	Runtimes[0] = totalTime;
	Runtimes[1] = tfrCPUtoGPU;
	Runtimes[2] = kernelExecutionTime;
	Runtimes[3] = tfrGPUtoCPU;
 
 	Error:
  for(i=0;i<2;i++) cudaStreamDestroy(stream[i]);
	cudaFree(GPU_odata);
	cudaFree(GPU_idata);
  cudaFree(Orientation);
  cudaFree(Gradient);
  cudaFree(GPU_CellHistogram);
  cudaFree(GPU_BlockHistogram);
	cudaFree(Displayhistogram);
  cudaFree(TempDisplayhistogram);
 	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
 
  return cudaStatus;
}
