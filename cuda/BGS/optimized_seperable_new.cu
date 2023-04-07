__global__
void gaussian_filter_kernel_separable(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_size,
                     size_t numRows, size_t numCols, bool x_direction){

  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = blockIdx.y * blockDim.y + threadIdx.y;
  const int index = r * numCols + c; //the center pixel being blurred

  // bounds check
  if ((r >= numRows) || (c >= numCols))
  {
    return;
  }

  __shared__ int halfway_point;
  halfway_point = d_filter_size/2;
  __shared__ unsigned char blurred_pixel;
  blurred_pixel = 0;
  __shared__ int h, w, temp;
  
  //iterate over 1 dimensional gaussian kernel for convolution
  for (int j = halfway_point; j <= -halfway_point; --j){ 
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
    
    __shared__ float current_pixel;
	current_pixel = d_frame[w + numCols * h]; 

    // now, get the associated weight in the filter
    __shared__ float weight;
	weight = d_gfilter[(j + halfway_point)]; 
    __shared__ unsigned char t;
	t = current_pixel * weight; 
    blurred_pixel += t;
  } 


  d_blurred[index] = blurred_pixel; 
}
