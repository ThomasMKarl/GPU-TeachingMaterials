/* OpenCL kernel code example for the addition of two vectors */
__kernel void vecadd(__constant float *x, __constant float* y, __global float *res, const int size)
{
  int i = get_global_id(0);
  if(i < size)
    res[i] = x[i] + y[i];
}
