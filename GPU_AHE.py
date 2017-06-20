import pycuda.autoinit
import pycuda.driver as drv
import numpy
from math import ceil

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void apply_kernel(unsigned char *res_img, unsigned char *img, int img_w, int img_h, int kernel_size, int max_intensity)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int kernel_size_squared = kernel_size*kernel_size;
  /*Check for boundary conditions*/
  if ((i + kernel_size >= img_w || j + kernel_size >= img_h) ) {
    if (i < img_w && j < img_h)
      res_img[i * img_h + j] = 0;
    return;
  }
	/*Rank calculation*/
  int rank = 0;
  const int curr_el = img[img_h * i + j];
  for (int curr_i = i; curr_i < i + kernel_size; ++curr_i)
    for (int curr_j = j; curr_j < j + kernel_size; ++curr_j) {
        if (img[curr_i * img_h + curr_j] < curr_el) {
            rank++;
        }
    }
  res_img[i * img_h + j] = (rank * max_intensity )/kernel_size_squared;
}
""")

def ahe(img, kernel_size, max_intensity):
    kernel = mod.get_function("apply_kernel")
    old_sh = img.shape
    img_w = img.shape[0]
    img_h = img.shape[1]
    block_size = (16,16,1)
    grid_size = (int(ceil(img_w * 1./block_size[0])), int(ceil(img_h * 1./block_size[1])))
    img.shape = -1
    res = numpy.zeros_like(img)
    kernel(drv.Out(res), drv.In(img), numpy.int32(img_w), numpy.int32(img_h), numpy.int32(kernel_size), numpy.int32(max_intensity), block=block_size, grid=grid_size)
    img.shape = old_sh
    res.shape = old_sh

    return res