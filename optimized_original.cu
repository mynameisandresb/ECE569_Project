__global__
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

  __shared__ int halfway_point;
  halfway_point = d_filter_width/2;
  __shared__ float blurred_pixel;
	blurred_pixel = 0.0f;
 
  numRows = numRows-1;
  numCols = numCols-1;
  // Iterate over 2D Gaussian kernel
  for (int i = -halfway_point; i <= halfway_point; ++i){ 
    for (int j = -halfway_point; j <= halfway_point; ++j){ 
            // get the location of the desired pixel, clamped to borders of the image
            int h = fmin(fmax((float)(r + i), 0.f), (float)(numRows)); 
            int w = fmin(fmax((float)(c + j), 0.f), (float)(numCols)); 
            int current_pixel_id = w + numCols * h;
            float current_pixel = static_cast<float>(d_frame[current_pixel_id]); 

            // now, get the associated weight in the filter
            current_pixel_id = (i + halfway_point) * d_filter_width + j + halfway_point; 
            float weight = d_gfilter[current_pixel_id]; 
            blurred_pixel += current_pixel * weight; 
        } 
    } 
 
  d_blurred[index] = static_cast<int>(blurred_pixel); 
}

