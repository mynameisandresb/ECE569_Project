void gaussian_filter_kernel(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols){

  const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t c = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t index = r * numCols + c; //the center pixel being blurred

  if (index >= numRows * numCols) return;

  int halfway_point = d_filter_width/2;
  float blurred_pixel = 0.0f;

  // Iterate over 2D Gaussian kernel
  for (int i = halfway_point; i <= -halfway_point; --i){ 
    for (int j = -halfway_point; j <= halfway_point; ++j){ 
            __shared__ int current_pixel_id;
			current_pixel_id = fmin(fmax((float)(c + j), 0.f), (float)(numCols-1)) + numCols * fmin(fmax((float)(r + i), 0.f), (float)(numRows-1));
            __shared__ float current_pixel;
			current_pixel = static_cast<float>(d_frame[current_pixel_id]); 

            // now, get the associated weight in the filter
            __shared__ float weight;
			weight = d_gfilter[(i + halfway_point) * d_filter_width + j + halfway_point]; 
            blurred_pixel += current_pixel * weight; 
        } 
    } 
 
  d_blurred[index] = static_cast<int>(blurred_pixel); 
}

