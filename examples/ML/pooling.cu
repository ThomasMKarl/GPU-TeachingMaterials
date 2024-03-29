/* Pooling with cuDNN*/
#include <cudnn.h>
#include <stdio.h>
#include <iostream>
#include <cmath>

#define IN_SIZE (2*2*10*10)
#define OUT_SIZE (2*2*8*8)
#define TOL (0.000005)

#define CUDNN_DTYPE CUDNN_DATA_FLOAT
typedef float stype;
typedef float dtype;

dtype input[IN_SIZE] = {0.262123f,-0.448813f,0.073700f,-0.144819f,2.388026f,-0.374544f,0.110159f,-1.542872f,-0.614891f,-2.051789f,0.549311f,-0.514576f,-0.359810f,-0.658335f,-0.187685f,0.648840f,-0.516337f,-0.868445f,0.362668f,1.031871f,-0.771410f,0.062409f,-0.374612f,-0.486497f,0.432054f,2.402000f,-0.441910f,2.352234f,0.581970f,-0.111883f,-0.888563f,0.514422f,0.561516f,-0.330782f,0.647885f,0.257522f,0.344512f,0.106130f,0.258680f,-1.026894f,1.269104f,1.062040f,-0.597162f,0.745507f,-0.760609f,0.687462f,-1.387311f,0.646558f,1.048010f,-1.010036f,-0.242760f,-0.886322f,0.923893f,0.274023f,0.307242f,-0.792433f,-0.890098f,-0.518272f,1.654387f,-0.030431f,-0.283656f,-0.821673f,0.226966f,-1.403059f,2.881252f,-0.177616f,-2.058162f,-0.038062f,-1.572661f,1.194749f,0.024840f,-0.288903f,-1.034757f,1.191242f,-0.151638f,0.766530f,-1.987274f,0.878733f,-0.044172f,-0.385836f,-2.814923f,0.328361f,-2.187914f,-0.652316f,0.665716f,1.840722f,0.444061f,-0.695111f,0.602564f,0.921938f,0.050023f,2.808196f,0.934590f,-0.120014f,1.330290f,0.461921f,-0.428371f,0.422131f,-1.090753f,-1.594703f,-0.696359f,1.504894f,-0.063938f,0.687890f,-0.421354f,1.489320f,-0.277929f,-1.389589f,-0.440915f,-0.837541f,-0.516718f,1.778166f,0.773082f,-0.383610f,1.718095f,-1.101422f,-1.461308f,0.612737f,-0.802261f,-0.690392f,0.264162f,2.597332f,0.000796f,0.998781f,-0.320811f,0.650900f,-1.959037f,1.344171f,-1.122830f,0.099737f,-1.079591f,-0.371666f,0.285432f,0.026553f,-0.008451f,0.030703f,-2.273837f,-0.304470f,3.024494f,-0.874619f,0.971874f,0.854032f,-0.479442f,0.747241f,0.999075f,0.009687f,2.001737f,0.077608f,0.102555f,0.617666f,-1.197479f,0.912035f,-0.408056f,0.131625f,-1.353883f,-0.813187f,2.142162f,-0.080583f,-0.069602f,-0.584452f,0.169271f,0.190131f,0.681481f,0.880227f,0.594960f,1.662274f,0.752301f,-0.796313f,0.159857f,0.335005f,0.362494f,-1.691298f,-0.460871f,0.478353f,-1.143353f,0.187721f,-1.552779f,0.501249f,-1.359241f,-0.262240f,-1.859266f,0.026011f,-0.123924f,-1.284480f,0.152293f,-0.778388f,-0.516877f,-0.029199f,0.034527f,-0.517577f,-0.946808f,-0.112903f,0.135832f,0.483324f,-1.489053f,-2.183046f,-0.157771f,-1.181278f,-0.193854f,0.131993f,-0.442095f,-2.041139f,0.968895f,-0.850359f,1.044780f,-0.033477f,2.244874f,1.504368f,-1.020596f,-0.663904f,1.477934f,-0.470869f,-1.203698f,-0.341503f,0.850112f,1.213592f,-0.352194f,0.491731f,0.671288f,3.116654f,0.069762f,0.227056f,0.122962f,0.480974f,0.300978f,1.302360f,-0.301896f,1.253229f,0.217611f,-0.422171f,0.708363f,-0.574906f,1.831306f,1.215541f,0.804596f,-0.353122f,1.085757f,-0.631121f,0.207719f,-1.233712f,-0.639785f,-0.497390f,0.481344f,0.758627f,-0.314814f,-0.129435f,0.305856f,1.437300f,-0.325752f,1.524121f,0.940245f,-2.035300f,-0.320925f,0.503123f,-0.551156f,-0.976267f,-1.018492f,-0.896719f,1.281112f,-0.756211f,1.597229f,-0.958358f,-0.632013f,-0.653434f,-1.622354f,0.507325f,-0.165485f,-0.982084f,-0.412180f,1.017206f,-0.355620f,-0.787981f,-0.668903f,1.595166f,0.799654f,-1.027516f,0.065277f,-0.592998f,-1.676015f,-0.524922f,0.056774f,1.104882f,-0.970155f,-0.891654f,1.430261f,-0.821645f,-0.487659f,0.752590f,-0.222564f,0.947882f,-2.292910f,-1.220162f,1.196011f,-0.568346f,1.348581f,-0.471481f,0.188347f,0.129570f,0.227159f,-0.236964f,0.196592f,1.008561f,-0.126765f,0.496254f,-0.948151f,0.528325f,0.246975f,-0.680574f,-2.046775f,-0.618394f,-0.514500f,-0.231317f,0.602324f,0.712511f,-1.214239f,-1.076851f,-0.757238f,-0.231837f,0.240245f,-0.294471f,-0.585524f,0.881918f,1.871276f,-0.707126f,-1.530920f,-1.619538f,-0.341249f,-0.482111f,1.538545f,0.435409f,0.749853f,-0.575038f,-1.457475f,-0.095831f,0.002680f,0.191423f,-0.735491f,0.273115f,-0.659371f,-0.010492f,-1.079095f,0.626513f,0.948138f,-0.696530f,-0.341514f,0.722822f,2.962597f,-1.163744f,-0.341071f,0.119832f,-0.244235f,-0.286361f,0.281030f,0.189141f,1.307471f,-1.079156f,1.774357f,-0.844799f,-0.407041f,0.061039f,-0.174228f,0.199437f,-0.024354f,-1.159870f,0.535349f,-0.260985f,-0.135743f,0.001018f,-1.634243f,2.377791f,0.450087f,-0.519948f,0.157897f,1.188245f,0.472171f,0.854651f,-0.018903f,0.673783f,0.027556f,-1.879497f,1.528776f,0.583272f,1.432598f,-0.500741f,-1.179192f,1.762375f,0.077588f,-0.346639f,0.618439f,0.121528f,0.253155f,0.826751f,0.930116f,-0.833282f,0.502162f,-0.009059f,-1.937293f,-0.393703f,1.918152f,0.655359f};

dtype output[OUT_SIZE] = {0.549311f,0.073700f,2.388026f,2.402000f,2.402000f,2.402000f,2.352234f,2.352234f,0.561516f,0.561516f,0.647885f,2.402000f,2.402000f,2.402000f,2.352234f,2.352234f,1.269104f,1.062040f,0.745507f,2.402000f,2.402000f,2.402000f,2.352234f,2.352234f,1.269104f,1.062040f,0.923893f,0.745507f,0.687462f,0.687462f,1.654387f,1.654387f,1.269104f,1.062040f,2.881252f,2.881252f,2.881252f,0.687462f,1.654387f,1.654387f,0.923893f,1.191242f,2.881252f,2.881252f,2.881252f,0.878733f,1.654387f,1.654387f,0.328361f,1.191242f,2.881252f,2.881252f,2.881252f,1.840722f,0.878733f,1.194749f,2.808196f,2.808196f,1.330290f,1.840722f,1.840722f,1.840722f,0.878733f,0.921938f,2.597332f,2.597332f,1.718095f,1.718095f,1.718095f,1.489320f,1.344171f,1.344171f,2.597332f,2.597332f,1.718095f,1.718095f,1.718095f,1.344171f,3.024494f,3.024494f,2.597332f,2.597332f,0.999075f,0.999075f,2.001737f,2.001737f,3.024494f,3.024494f,0.971874f,0.912035f,0.999075f,0.999075f,2.142162f,2.142162f,3.024494f,3.024494f,0.971874f,0.912035f,0.999075f,1.662274f,2.142162f,2.142162f,2.142162f,0.617666f,0.912035f,0.912035f,0.880227f,1.662274f,2.142162f,2.142162f,2.142162f,0.501249f,0.681481f,0.880227f,0.880227f,1.662274f,1.662274f,1.662274f,0.752301f,0.501249f,0.362494f,0.483324f,0.483324f,0.483324f,0.187721f,0.501249f,0.501249f,0.501249f,1.477934f,0.968895f,1.044780f,1.302360f,2.244874f,2.244874f,2.244874f,3.116654f,1.831306f,1.831306f,1.831306f,1.302360f,1.302360f,1.302360f,1.253229f,3.116654f,1.831306f,1.831306f,1.831306f,1.302360f,1.302360f,1.437300f,1.437300f,1.524121f,1.831306f,1.831306f,1.831306f,1.215541f,1.085757f,1.437300f,1.437300f,1.524121f,1.597229f,0.758627f,0.758627f,0.758627f,0.507325f,1.437300f,1.437300f,1.524121f,1.597229f,1.595166f,1.595166f,1.595166f,0.799654f,0.507325f,1.281112f,1.281112f,1.597229f,1.595166f,1.595166f,1.595166f,1.430261f,0.752590f,0.752590f,1.017206f,1.196011f,1.595166f,1.595166f,1.595166f,1.430261f,0.752590f,0.752590f,0.947882f,1.871276f,1.871276f,1.871276f,0.712511f,0.528325f,0.528325f,1.538545f,1.538545f,1.871276f,1.871276f,1.871276f,0.712511f,0.191423f,0.273115f,1.538545f,1.538545f,1.871276f,1.871276f,1.871276f,0.722822f,2.962597f,2.962597f,2.962597f,1.538545f,0.948138f,0.948138f,1.307471f,1.307471f,2.962597f,2.962597f,2.962597f,0.273115f,0.948138f,0.948138f,1.307471f,1.307471f,2.962597f,2.962597f,2.962597f,2.377791f,0.450087f,1.188245f,1.307471f,1.307471f,1.774357f,1.774357f,1.774357f,2.377791f,1.528776f,1.432598f,1.432598f,1.762375f,1.762375f,1.762375f,0.673783f,2.377791f,1.528776f,1.432598f,1.432598f,1.762375f,1.762375f,1.762375f,1.918152f,1.918152f};

dtype gradient[IN_SIZE] = {0.000000f,0.000000f,0.073700f,0.000000f,2.388026f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.549311f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,21.618002f,0.000000f,14.113403f,0.000000f,0.000000f,0.000000f,0.000000f,1.123032f,0.000000f,0.647885f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,3.807312f,3.186121f,0.000000f,1.491014f,0.000000f,2.062386f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.847786f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,9.926323f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,25.931272f,0.000000f,0.000000f,0.000000f,0.000000f,1.194749f,0.000000f,0.000000f,0.000000f,2.382483f,0.000000f,0.000000f,0.000000f,2.636199f,0.000000f,0.000000f,0.000000f,0.328361f,0.000000f,0.000000f,0.000000f,7.362889f,0.000000f,0.000000f,0.000000f,0.921938f,0.000000f,5.616391f,0.000000f,0.000000f,1.330290f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.489320f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,10.308568f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,15.583992f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,4.032512f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,18.146962f,0.000000f,1.943747f,0.000000f,0.000000f,0.000000f,4.995373f,0.000000f,4.003474f,0.000000f,0.000000f,0.617666f,0.000000f,3.648140f,0.000000f,0.000000f,0.000000f,0.000000f,17.137293f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.681481f,2.640680f,0.000000f,8.311371f,0.752301f,0.000000f,0.000000f,0.000000f,0.362494f,0.000000f,0.000000f,0.000000f,0.000000f,0.187721f,0.000000f,2.506245f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.449971f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.968895f,0.000000f,1.044780f,0.000000f,6.734622f,0.000000f,0.000000f,0.000000f,1.477934f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,6.233308f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,7.814159f,0.000000f,1.253229f,0.000000f,0.000000f,0.000000f,0.000000f,16.481757f,1.215541f,0.000000f,0.000000f,1.085757f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,2.275881f,0.000000f,0.000000f,0.000000f,8.623803f,0.000000f,4.572364f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,2.562223f,0.000000f,4.791688f,0.000000f,0.000000f,0.000000f,0.000000f,1.014651f,0.000000f,0.000000f,0.000000f,1.017206f,0.000000f,0.000000f,0.000000f,14.356496f,0.799654f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,2.860521f,0.000000f,0.000000f,3.010359f,0.000000f,0.947882f,0.000000f,0.000000f,1.196011f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.056650f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,1.425023f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,16.841485f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,7.692725f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.191423f,0.000000f,0.546231f,0.000000f,0.000000f,0.000000f,0.000000f,3.792553f,0.000000f,0.000000f,0.722822f,26.663373f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,7.844826f,0.000000f,5.323071f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,7.133373f,0.450087f,0.000000f,0.000000f,1.188245f,0.000000f,0.000000f,0.000000f,0.673783f,0.000000f,0.000000f,3.057553f,0.000000f,5.730390f,0.000000f,0.000000f,10.574248f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,0.000000f,3.836303f,0.000000f};

#define IN_DATA_BYTES (IN_SIZE*sizeof(dtype))
#define OUT_DATA_BYTES (OUT_SIZE*sizeof(dtype))

//function to print out error message from cuDNN calls
#define checkCUDNN(exp) \
  { \
    cudnnStatus_t status = (exp); \
    if(status != CUDNN_STATUS_SUCCESS) { \
      std::cerr << "Error on line " << __LINE__ << ": " \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE); \
    } \
  } 

int main()
{
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnPoolingDescriptor_t pooling_desc;
  //create descriptor handle
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  //initialize descriptor
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         CUDNN_POOLING_MAX,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         3,                       //window height
                                         3,                       //window width
                                         0,                       //vertical padding
                                         0,                       //horizontal padding
                                         1,                       //vertical stride
                                         1));                     //horizontal stride
  
  cudnnTensorDescriptor_t in_desc;
  //create input data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
  //initialize input data descriptor 
  checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,                  //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        2,                        //number of images
                                        2,                        //number of channels
                                        10,                       //data height 
                                        10));                     //data width

  cudnnTensorDescriptor_t out_desc;
  //create output data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
  //initialize output data descriptor
  checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,                 //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DTYPE,              //data type (precision)
                                        2,                        //number of images
                                        2,                        //number of channels
                                        8,                        //data height
                                        8));                      //data width

  stype alpha = 1.0f;
  stype beta = 0.0f;
  //GPU data pointers
  dtype *in_data, *out_data;
  //allocate arrays on GPU
  cudaMalloc(&in_data,IN_DATA_BYTES);
  cudaMalloc(&out_data,OUT_DATA_BYTES);
  //copy input data to GPU array
  cudaMemcpy(in_data,input,IN_DATA_BYTES,cudaMemcpyHostToDevice);
  //initize output data on GPU
  cudaMemset(out_data,0,OUT_DATA_BYTES);

  //Call pooling operator
  checkCUDNN(cudnnPoolingForward(cudnn,         //cuDNN context handle
                                 pooling_desc,  //pooling descriptor handle
                                 &alpha,        //alpha scaling factor
                                 in_desc,       //input tensor descriptor
                                 in_data,       //input data pointer to GPU memory
                                 &beta,         //beta scaling factor
                                 out_desc,      //output tensor descriptor
                                 out_data));    //output data pointer from GPU memory

  //allocate array on CPU for output tensor data
  dtype *result = (dtype*)malloc(OUT_DATA_BYTES);
  //copy output data from GPU
  cudaMemcpy(result,out_data,OUT_DATA_BYTES,cudaMemcpyDeviceToHost);

  //loop over and check that the forward pass outputs match expected results (exactly)
  int err = 0;
  for(int i=0; i<OUT_SIZE; i++) {
    if(result[i] != output[i]) {
      std::cout << "Error! Expected " << output[i] << " got " << result[i] << " for idx " << i <<std::endl;
      err++;
    }
  }

  std::cout << "Forward finished with " << err << " errors" << std::endl;

  dtype *in_grad;
  //allocate array on GPU for gradient
  cudaMalloc(&in_grad,IN_DATA_BYTES);
  //initialize output array 
  cudaMemset(in_grad,0,IN_DATA_BYTES);

  //call pooling operator to compute gradient
  checkCUDNN(cudnnPoolingBackward(cudnn,        //cuDNN context handle
                                  pooling_desc, //pooling descriptor handle
                                  &alpha,       //alpha scaling factor
                                  out_desc,     //output tensor descriptor
                                  out_data,     //output tensor pointer to GPU memory
                                  out_desc,     //differential tensor descriptor
                                  out_data,     //differential tensor pointer to GPU memory
                                  in_desc,      //input tensor descriptor
                                  in_data,      //input tensor pointer to GPU memory
                                  &beta,        //beta scaling factor
                                  in_desc,      //gradient tensor descriptor
                                  in_grad));    //gradient tensor pointer to GPU memory

  //allocate array on CPU for gradient tensor data
  dtype *grad = (dtype*)malloc(IN_DATA_BYTES);
  //copy gradient data from GPU
  cudaMemcpy(grad,in_grad,IN_DATA_BYTES,cudaMemcpyDeviceToHost);

  //loop over and check that the forward pass outputs match expected results (within tolerance)
  err = 0;
  for(int i=0; i<IN_SIZE; i++) {
    double diff = std::abs(gradient[i] - grad[i]);
    if(diff > TOL) {
      std::cout << "Error! Expected " << gradient[i] << " got " << grad[i] << " for idx " << i << " diff: " << diff <<std::endl;
      err++;
    }
  }

  std::cout << "Backward finished with " << err << " errors" << std::endl;

  //free CPU arrays
  free(result);
  free(grad);

  //free GPU arrays
  cudaFree(in_data);
  cudaFree(in_grad);
  cudaFree(out_data);

  //free cuDNN descriptors
  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  cudnnDestroy(cudnn);
  
  return 0;
}
