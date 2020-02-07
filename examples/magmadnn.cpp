/* example in MAGMADNN for MNIST image classification with a simple neural network */
#include  <vector>
#include  <utility>
#include  "magmadnn.h"
#include  "utils/mnist_parser/mnist_image_parser.h"

using  namespace  MagmaDNN;

int  main(int  argc, char** argv)
{
    if (argc != 7)
    {
        printf("usage: %s image_data label_data epochs batches learning_rate weight_decay\n", argv[0]);
        exit(1);
    }
    
    // read in the input data into c-matrices
    float*** f_raw_mnist = NULL;
    int  n_images, n_rows, n_cols;
    read_mnist_image(argv[1], &f_raw_mnist, &n_images, &n_rows, &n_cols);
    int* i_labels_mnist = NULL;
    int  n_labels;
    read_mnist_label(argv[2], &i_labels_mnist, &n_labels);
    
    // initialize magma
    magma_init();
    
    magma_print_environment();
    
    magma_int_t  n_batch = std::stoi(argv[4]);
    magma_int_t  n_features = n_rows * n_cols;
    magma_int_t  n_output_classes = 10;
    Tensor<float> x_train ({n_images, n_features});
    Tensor<float> y_train ({n_images, n_output_classes});
    Tensor<float> x_batch ({n_batch, n_features});
    Tensor<float> y_batch ({n_batch, n_output_classes});
    
    // read MNIST data into tensors and free memory
    for (int  i = 0; i < n_images; i++)
    {
        for (int  j = 0; j < n_rows; j++)
	{
            for (int  k = 0; k < n_cols; k++)
	    {
                x_train.set_by_idx(i, (j * n_rows + k), (f_raw\_mnist[i][j][k] / 128.0)-1.0);
            }
        }
        y_train.set_by_idx(i, i_labels_mnist[i], 1.0);
    }
    FREE_3D_ARRAY(f_raw_mnist, n_images, n_rows);
    free(i_labels_mnist);
    
    // create the network layers
    InputLayer<float>       input_layer                                (x_batch);
    FCLayer<float>*         FC1          = new  FCLayer<float>         (&input_layer, 512);
    ActivationLayer<float>* actv1        = new  ActivationLayer<float> (FC1, SIGMOID);
    FCLayer<float>*         FC2          = new  FCLayer<float>         (actv1, 10);
    ActivationLayer<float>* actv2        = new  ActivationLayer<float> (FC2, RELU);
    OutputLayer<float>*     output_layer = new  OutputLayer<float>     (actv2, y_batch, BIN_CROSSENTROPY_WITH_SIGMOID);
    std::vector<Layer<float>*> layers { &input_layer, FC1, actv1, FC2, actv2, output_layer };
    Param  p {std::stof(argv[5]), std::stof(argv[6]), n_batch, std::stoi(argv[3])};
    Model  model (p, &layers);
    // training tensors, verbose, and return pair (acc, loss)
    model.fit(x_train, y_train, true, nullptr);
    
    magma_finalize();
    
    return  0;
}
