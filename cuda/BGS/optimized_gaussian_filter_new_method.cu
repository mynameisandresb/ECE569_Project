__global__
void gaussian_filter_kernel(unsigned char* d_frame,
                     unsigned char* d_blurred,
                     const float* const d_gfilter,
                     size_t d_filter_width,
                     size_t d_filter_height,
                     size_t numRows, size_t numCols){
	
   #define BLOCK_SIZE 16
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x >= d_filter_width || y >= d_filter_height) {
       return;
  }

   float sum = 0.0f;

   float s_data[BLOCK_SIZE + 9 - 1][BLOCK_SIZE + 9- 1];

   int x0 = blockIdx.x * blockDim.x - (d_filter_width - 1) / 2;
   int y0 = blockIdx.y * blockDim.y - (d_filter_width - 1) / 2;
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   s_data[ty][tx] = d_blurred[(y0 + ty) * d_filter_width + x0 + tx];

    if (tx < d_filter_width - 1) {
        s_data[ty][tx + BLOCK_SIZE] = d_blurred[(y0 + ty) * d_filter_width + x0 + tx + BLOCK_SIZE];
    }

    if (ty < d_filter_width - 1) {
        s_data[ty + BLOCK_SIZE][tx] = d_blurred[(y0 + ty + BLOCK_SIZE) * d_filter_width + x0 + tx];
    }

    if (tx < d_filter_width - 1 && ty < d_filter_width - 1) {
        s_data[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = d_blurred[(y0 + ty + BLOCK_SIZE) * d_filter_width + x0 + tx + BLOCK_SIZE];
    }

    __syncthreads();

    // Compute convolution
    for (int i = 0; i < d_filter_width; i++) {
        for (int j = 0; j < d_filter_width; j++) {
            sum += s_data[ty + i][tx + j] * d_filter[i * d_filter_width + j];
        }
    }

    d_frame[y * 9 + x] = sum;

}

