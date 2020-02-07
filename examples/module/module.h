//Header for module.cu
#ifndef MODULE_H
#define MODULE_H

#include<stdio.h>
#include "../gpuerror.h"

void test(float*);
__global__ void kernel(float*);

#endif
