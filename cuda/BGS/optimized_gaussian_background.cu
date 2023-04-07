__global__
void gaussian_background_kernel(unsigned char * const d_frame,
                            unsigned char* const d_amean, 
                            unsigned char* const d_cmean,
                            unsigned char* const d_avar,
                            unsigned char* const d_cvar,
                            unsigned char* const d_bin,
                            int * const d_aage,
                            int * const d_cage,
                       int numRows, int numCols)
{
  const size_t r = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t c = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t index = r * numCols + c;

  // Make sure you're in memory bounds
  if (index < numRows * numCols)
  {

    __shared__ float alpha, V;
    __shared__ int adiff;
    __shared__ int cdiff;
  // Adjust memory handling, utilize shared memory
    __shared__ float pixel; 
	pixel = d_frame[index];
    __shared__ float ameanpixel;
	ameanpixel = d_amean[index];
    __shared__ float avarpixel;
 	avarpixel = d_avar[index];
    __shared__ float cmeanpixel;
 	cmeanpixel= d_cmean[index];
    __shared__ float cvarpixel;
    cvarpixel = d_cvar[index];

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
