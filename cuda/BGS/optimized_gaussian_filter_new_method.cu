__global__
void gaussian_filter_kernel(unsigned char* d_frame,
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
	const int BLOCK_SIZE = 16;
	const int FILTER_WIDTH = d_filter_width;
	const int FILTER_HEIGHT = d_filter_height;

	__shared__ float s_data[BLOCK_SIZE + 9 - 1][BLOCK_SIZE + 9 - 1];

	if(row < numRows && col < numCols){
		s_data[ty][tx] = d_blurred[row * numCols + col];
	}

	if (tx < FILTER_WIDTH - 1 && col + BLOCK_SIZE < numCols) {
		s_data[ty][tx + BLOCK_SIZE] = d_blurred[row * numCols + col + BLOCK_SIZE];
	}

	if (ty < FILTER_HEIGHT - 1 && row + BLOCK_SIZE < numRows) {
		s_data[ty + BLOCK_SIZE][tx] = d_blurred[(row + BLOCK_SIZE) * numCols + col];
	}

	if (tx < FILTER_WIDTH - 1 && ty < FILTER_HEIGHT - 1 && col + BLOCK_SIZE < numCols && row + BLOCK_SIZE < numRows) 	{
		s_data[ty + BLOCK_SIZE][tx + BLOCK_SIZE] = d_blurred[(row + BLOCK_SIZE) * numCols + col + BLOCK_SIZE];
	}

	__syncthreads();

	float sum = 0.0f;
	#pragma unroll
	for (int i = 0; i < FILTER_HEIGHT; i++) {
		#pragma unroll		
		for (int j = 0; j < FILTER_WIDTH; j++) {
		    sum += s_data[ty + i][tx + j] * d_gfilter[i * FILTER_WIDTH + j];
		}
	}

	if (row < numRows && col < numCols) {
		d_frame[row * numCols + col] = sum;
	}
}
