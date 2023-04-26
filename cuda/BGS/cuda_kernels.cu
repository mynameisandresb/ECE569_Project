#include "utils.h"

/*
* Specify the size of a THREAD BLOCK of sizze THREAD_SIZE x THREAD_SIZE
*/
#define THREAD_SIZE 11
/*
* Flag whether to use separable gaussian filter or 2D
*/
#define SEPARATED_GAUSSIAN_FILTER 0

/**
* CUDA Kernel for DSGM
*/
__global__
void gaussian_background_kernel_opt(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                       int numRows, int numCols)
{
  const size_t r = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t c = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t index = r * numCols + c;

  // Make sure you're in memory bounds
  if (index < numRows * numCols)
  {

    float alpha, V;
    int adiff;
    int cdiff;

    float pixel = d_frame[index];
    float ameanpixel = d_amean[index];
    float avarpixel = d_avar[index];
    float cmeanpixel = d_cmean[index];
    float cvarpixel = d_cvar[index];

    adiff = pixel - ameanpixel;
    cdiff = pixel - cmeanpixel;
    // If within some threshold of the absolute background, update
    if(adiff*adiff < 9 * avarpixel){
        alpha = 1.0f / (float)d_aage[index];
        d_amean[index] = (1.0f-alpha) * ameanpixel + (alpha) * pixel;
        adiff = d_amean[index] - pixel;
        V = adiff*adiff;
        d_avar[index] = (1.0f-alpha) * avarpixel + alpha * V;
        d_aage[index]++;
    }
    //otherwise if in some threshold of the candidate, update
    else if(cdiff*cdiff < 9 * cvarpixel){
        alpha = 1.0f / (float)d_cage[index];
        d_cmean[index] = (1.0f-alpha) * cmeanpixel + (alpha) * pixel;
        cdiff = d_cmean[index] - pixel;
        V = cdiff*cdiff;
        d_cvar[index] = (1.0f-alpha) * cvarpixel + alpha * V;
        d_cage[index]++;
    }
    //otherwise reset candidate
    else{      
        d_cmean[index] = pixel;
        d_cvar[index] = 255;
        d_cage[index] = 1;
    }

    //if candidate age is larger
    if(d_cage[index] > d_aage[index]){
      //swap the candidate to the absolute
      d_amean[index] = d_cmean[index];
      d_avar[index] = d_cvar[index];
      d_aage[index] = d_cage[index];

      //reset the candidate model
      d_cmean[index] = pixel;
      d_cvar[index] = 255;
      d_cage[index] = 1;
    }

    adiff = pixel - d_amean[index];

    // Update motion mask
    if (adiff*adiff <= 60) {
        //background
        d_bin[index]= 0;
    } else {
        //foreground
        d_bin[index] = 255;
    }
  }
}

/**
* A call to the CUDA kernel, specify the block size first
*/
void gaussian_background_opt(unsigned char* const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                            size_t numRows, size_t numCols)
{
  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numCols / THREAD_SIZE + 1, numRows / THREAD_SIZE + 1, 1); 
  gaussian_background_kernel_opt<<<gridSize, blockSize>>>(d_frame, d_amean, d_cmean, 
                                              d_avar, d_cvar, d_bin, d_aage, d_cage,
                                                numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}

__global__
void gaussian_filter_kernel_opt(unsigned char* d_frame,
							unsigned char* d_blurred,
							const float* const d_gfilter,
							const int d_filter_width,
							const int d_filter_height,
							const int numRows, const int numCols){

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x;
	const int by = blockIdx.y;
	const int row = by * blockDim.y + ty;
	const int col = bx * blockDim.x + tx;
	const int FILTER_WIDTH = d_filter_width;
	const int FILTER_HEIGHT = d_filter_height;

	__shared__ float s_data[THREAD_SIZE + 9 - 1][THREAD_SIZE + 9 - 1];

	if(row < numRows && col < numCols){
		s_data[ty][tx] = d_frame[row * numCols + col];
	}

	if (tx < FILTER_WIDTH - 1 && col + THREAD_SIZE < numCols) {
		s_data[ty][tx + THREAD_SIZE] = d_frame[row * numCols + col + THREAD_SIZE];
	}

	if (ty < FILTER_HEIGHT - 1 && row + THREAD_SIZE < numRows) {
		s_data[ty + THREAD_SIZE][tx] = d_frame[(row + THREAD_SIZE) * numCols + col];
	}

	if (tx < FILTER_WIDTH - 1 && ty < FILTER_HEIGHT - 1 && col + THREAD_SIZE < numCols && row + THREAD_SIZE < numRows) 	{
		s_data[ty + THREAD_SIZE][tx + THREAD_SIZE] = d_frame[(row + THREAD_SIZE) * numCols + col + THREAD_SIZE];
	}

	__syncthreads();

  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < FILTER_HEIGHT; i++) {
    #pragma unroll		
    for (int j = 0; j < FILTER_WIDTH; j++) {
        float current_pixel = static_cast<float>(s_data[ty+i][tx+j]);
        //float current_pixel = static_cast<float>(d_frame[(row + i) * numCols + col + j];);
        sum += current_pixel * d_gfilter[i * FILTER_WIDTH + j];
    }
  }
	if (row < (numRows - 7) && col < (numCols - 7)) {
		d_blurred[row * numCols + col] = static_cast<int>(sum);
	}
}

/**
* Shared Memory Median filter CUDA kernel
* NOTE: This method is from paper High Performance Median Filtering Algorithm Based on NVIDIA GPU Computing
*/
__global__ void median_filter_kernel_v2(unsigned char* d_frame,
                                            unsigned char* d_blurred,
                                            size_t numRows, size_t numCols)
{
	//Set the row and col value for each thread.
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ unsigned char tile[(THREAD_SIZE+2)][(THREAD_SIZE+2)];  //initialize shared memory

	// Declares the boundary conditions for the shared memory
	bool is_x_left = (threadIdx.x == 0);
  bool is_x_right = (threadIdx.x == THREAD_SIZE-1);
  bool is_y_top = (threadIdx.y == 0);
  bool is_y_bottom = (threadIdx.y == THREAD_SIZE-1);

	//Initialize with zero
	if(is_x_left)
		tile[threadIdx.x][threadIdx.y+1] = 0;
	else if(is_x_right)
		tile[threadIdx.x + 2][threadIdx.y+1]=0;
	if (is_y_top){
		tile[threadIdx.x+1][threadIdx.y] = 0;
		if(is_x_left)
			tile[threadIdx.x][threadIdx.y] = 0;
		else if(is_x_right)
			tile[threadIdx.x+2][threadIdx.y] = 0;
	}
	else if (is_y_bottom){
		tile[threadIdx.x+1][threadIdx.y+2] = 0;
		if(is_x_right)
			tile[threadIdx.x+2][threadIdx.y+2] = 0;
		else if(is_x_left)
			tile[threadIdx.x][threadIdx.y+2] = 0;
	}

	//Setup pixel values
	tile[threadIdx.x+1][threadIdx.y+1] = d_frame[row*numCols+col];
	//Check for boundry conditions.
	if(is_x_left && (col>0))
		tile[threadIdx.x][threadIdx.y+1] = d_frame[row*numCols+(col-1)];
	else if(is_x_right && (col<numCols-1))
		tile[threadIdx.x + 2][threadIdx.y+1]= d_frame[row*numCols+(col+1)];
	if (is_y_top && (row>0)){
		tile[threadIdx.x+1][threadIdx.y] = d_frame[(row-1)*numCols+col];
		if(is_x_left)
			tile[threadIdx.x][threadIdx.y] = d_frame[(row-1)*numCols+(col-1)];
		else if(is_x_right )
			tile[threadIdx.x+2][threadIdx.y] = d_frame[(row-1)*numCols+(col+1)];
	}
	else if (is_y_bottom && (row<numRows-1)){
		tile[threadIdx.x+1][threadIdx.y+2] = d_frame[(row+1)*numCols + col];
		if(is_x_right)
			tile[threadIdx.x+2][threadIdx.y+2] = d_frame[(row+1)*numCols+(col+1)];
		else if(is_x_left)
			tile[threadIdx.x][threadIdx.y+2] = d_frame[(row+1)*numCols+(col-1)];
	}

	__syncthreads();   //Wait for all threads to be done.

	//Setup the filter.
	unsigned char filterVector[9] = {tile[threadIdx.x][threadIdx.y], tile[threadIdx.x+1][threadIdx.y], tile[threadIdx.x+2][threadIdx.y],
                   tile[threadIdx.x][threadIdx.y+1], tile[threadIdx.x+1][threadIdx.y+1], tile[threadIdx.x+2][threadIdx.y+1],
                   tile[threadIdx.x] [threadIdx.y+2], tile[threadIdx.x+1][threadIdx.y+2], tile[threadIdx.x+2][threadIdx.y+2]};

	
	{
		for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (filterVector[i] > filterVector[j]) { 
				        //Swap Values.
                char tmp = filterVector[i];
                filterVector[i] = filterVector[j];
                filterVector[j] = tmp;
            }
        }
    }
	d_blurred[row*numCols+col] = filterVector[4];
	}
}

/**
* Median filter CUDA kernel
* NOTE: parts of this were taken from: http://stackoverflow.com/questions/19634328/2d-cuda-median-filter-optimization
*/
__global__
void median_filter_kernel_v1(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols){

    const int size = 9;
    // unsigned short surround[9];

    int iterator, i;

    const int x = blockDim.y * blockIdx.y + threadIdx.y;
    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int index = x * numCols + y;

    // if out of bounds return
    if( (x >= (numRows)) || (y >= numCols) || (x < 0) || (y < 0)) return;

    //if border, don't blur
    if( (x == (numRows - 1)) || (y == numCols - 1) || (x == 0) || (y == 0)){
      d_blurred[index] = d_frame[index];
      return;
    }

    //Setup the filter.
    unsigned char filterVector[9] = {d_frame[x*numCols + y], d_frame[x*numCols + 1+y], d_frame[x*numCols + 2+y],
                    d_frame[x*numCols + y+1], d_frame[(x*numCols+1) + y+1], d_frame[(x*numCols+2) + y+1],
                    d_frame[x*numCols + y+2], d_frame[(x*numCols+1) + y+2], d_frame[(x*numCols+2) + y+2]};

    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (filterVector[i] > filterVector[j]) { 
                //Swap Values.
                char tmp = filterVector[i];
                filterVector[i] = filterVector[j];
                filterVector[j] = tmp;
            }
        }
    }
    d_blurred[x*numCols+y] = filterVector[4];
}

/**
* Median filter CUDA kernel
* NOTE: parts of this were taken from: http://stackoverflow.com/questions/19634328/2d-cuda-median-filter-optimization
*/
__global__
void median_filter_kernel_v0(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols){

    const int size = 9;
    unsigned short surround[9];

    int iterator, i;

    const int x = blockDim.y * blockIdx.y + threadIdx.y;
    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    const int index = x * numCols + y;

    // if out of bounds return
    if( (x >= (numRows)) || (y >= numCols) || (x < 0) || (y < 0)) return;

    //if border, don't blur
    if( (x == (numRows - 1)) || (y == numCols - 1) || (x == 0) || (y == 0)){
      d_blurred[index] = d_frame[index];
      return;
    }

    // get the surrounding pixels and fill a local array
    iterator = 0;
    for (int r = x - 1; r <= x + 1; r++) {
        for (int c = y - 1; c <= y + 1; c++) {
            surround[iterator] = d_frame[r*numCols+c];
            iterator++;
        }
    }


    // simple sorting
    int middle = (size/2)+1;
    for (i=0; i<=middle; i++) {
        int minval=i;
        for (int l=i+1; l<size; l++){
          if (surround[l] < surround[minval]){
             minval=l;
          }
        }
        unsigned short temp = surround[i];
        surround[i]=surround[minval];
        surround[minval]=temp;
    }

    // Set to the median value
    d_blurred[index] = surround[middle]; 
}


/**
* Call to the gaussian filter CUDA kernel to specify the blocksize
*/
void gaussian_filter_opt(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numCols / THREAD_SIZE + 1, numRows / THREAD_SIZE + 1, 1); 
  gaussian_filter_kernel_opt<<<gridSize, blockSize>>>(d_frame, d_blurred, d_gfilter, 
                                                  d_filter_width, d_filter_height, 
                                                  numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* A separable gaussian filter kernel
*/
__global__
void gaussian_filter_kernel_separable_opt(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols, bool x_direction){

  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = r * numCols + c; //the center pixel being blurred

  // bounds check
  if ((r >= numRows) || (c >= numCols))
  {
    return;
  }

  int halfway_point = d_filter_size/2;
  unsigned char blurred_pixel = 0;
  int h, w, temp;
  
  //iterate over 1 dimensional gaussian kernel for convolution
  for (int j = -halfway_point; j <= halfway_point; ++j){ 
    //get the desired direction and clamp to borders    
    if(x_direction){
      temp = r+j;
      if(temp > numRows-1) temp = numRows-1;
      else if(temp < 0) temp = 0;
      h = temp; 
      w = c;
    }
    else{
      temp = c+j;
      if(temp > numCols-1) temp = numCols-1;
      else if(temp < 0) temp = 0;

      w = temp; 
      h = r;
    }
    
    size_t current_pixel_id = w + numCols * h;
    float current_pixel = d_frame[current_pixel_id]; 

    // now, get the associated weight in the filter
    current_pixel_id = (j + halfway_point); 
    float weight = d_gfilter[current_pixel_id]; 
    unsigned char t = current_pixel * weight; 
    blurred_pixel += t;
  } 


  d_blurred[index] = blurred_pixel; 
}

/**
* Call to the separable filter
*/
void gaussian_filter_separable_opt(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numCols / THREAD_SIZE + 1, numRows / THREAD_SIZE + 1, 1); 
  // once in the x direction
  gaussian_filter_kernel_separable_opt<<<gridSize, blockSize>>>(d_frame, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, true);
  //once in the y
  gaussian_filter_kernel_separable_opt<<<gridSize, blockSize>>>(d_blurred_temp, d_blurred, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, false);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* Call to the median filter
*/
void median_filter_opt(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numCols / THREAD_SIZE + 1, numRows / THREAD_SIZE + 1, 1); 
  // once in the x direction
  //default median filter size 3
  median_filter_kernel_v1<<<gridSize, blockSize>>>(d_frame, d_blurred, numRows, numCols);

  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/**
* A sequenced call to either the separable gaussian filter or the 2d filter and a subsequent call
* to the median filter CUDA kernels to run on the GPU with the device memory pointers provided
*/
void gaussian_and_median_blur_opt(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     unsigned char* d_blurred_temp,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols)
{

  const dim3 blockSize(THREAD_SIZE, THREAD_SIZE, 1);
  const dim3 gridSize(numCols / THREAD_SIZE + 1, numRows / THREAD_SIZE + 1, 1); 

  #if SEPARATED_GAUSSIAN_FILTER == 1
  // once in the x direction
  gaussian_filter_kernel_separable_opt<<<gridSize, blockSize>>>(d_frame, d_blurred, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, true);

  //once in the y direction
  gaussian_filter_kernel_separable_opt<<<gridSize, blockSize>>>(d_blurred, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, 
                                                  numRows, numCols, false);
  #else
  // in this case, also need to make sure the filter is 2d
  gaussian_filter_kernel_opt<<<gridSize, blockSize>>>(d_frame, d_blurred_temp, d_gfilter, 
                                                  d_filter_size, d_filter_size, 
                                                  numRows, numCols);
  #endif

  median_filter_kernel_v1<<<gridSize, blockSize>>>(d_blurred_temp, d_blurred, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}