//https://www.open-mpi.org/faq/?category=mpi-apps
//mpic++ --showme:link
//nvcc -O3 ising_cluster.cpp -lmpi -lcudart -L /usr/local/cuda/lib64 -L/usr/lib/openmpi -lmpi_cxx -lmpi -o ising_cluster
//mpiexec -H ch200,ch201,ch202,ch203,ch204,ch205,ch206,ch207,ch208,ch209 ./ising_cluster
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

template <typename T>
T mean(std::vector<T> &o)
{
    size_t n = o.size();
    
    T meanvalue = 0.0;
    #pragma omp parallel for reduction(+:meanvalue)
    for(size_t i = 0; i < n; ++i) meanvalue += o[i];
    
    return meanvalue/T(n);
}

template <typename T>
T covFunc(size_t t, std::vector<T> &o) //autocovariance-function
{
    size_t n = o.size() - t;
    
    T a = 0.0;
    T b = 0.0;
    T c = 0.0;

    #pragma omp parallel for reduction(+:a,b,c)
    for(size_t i = 0; i < n; ++i)
    {
        a += o[i]*o[i+t];
        b += o[i];
        c += o[i+t];
    } 
    
    a /= T(n);
    b /= T(n);
    c /= T(n);
 
    return a-(b*c);
}

template <typename T>
T intAuto(std::vector<T> &o) //integrated autocorrelation-time
{
    size_t n = o.size();

    T sum = 0;
    T var = covFunc(0, o);
    
    T cov;
    for(size_t t = 1; t < n; t++)
    {
        cov = covFunc(t, o);
        if(cov > 0)
        {
	    sum += (1-t/T(n)) * cov/var;
        }
        else break;
    }

    return 0.5 + sum;
}

template <typename T>
T error(std::vector<T> &o) //error on expectation for given observable
{
    return sqrt(covFunc(0, o)/T(o.size()));
}
template <typename T>
size_t d2i(T d) //round double to int
{
    if(d < 0) d *= -1;
    return d<0?d-.5:d+.5;
}

template <typename T>
void removeCorr(std::vector<T> &o)
{
    std::vector<T> help;
    T autot = intAuto(o);
    size_t n = o.size();
    size_t therm = d2i(20*autot);
    size_t corr;

    if(therm >= n)
    {
        std::cerr << n << " values given, but thermalisation needed " << therm << " steps!" << "\n";
        return;
    }   
    //std::cout << therm << std::endl;

    n -= therm;
    help.resize(n);
    #pragma omp parallel for
    for(size_t i = 0; i < n; ++i) help[i] = o[i+therm];
        
    autot = intAuto(o);
    //std::cout << autot << std::endl;
    
    corr = d2i(2.0*autot);
    if(corr == 1) return;
    n /= corr;
    
    o.resize(n);
    #pragma omp parallel for
    for(size_t i = 0; i < n; ++i) o[i] = help[i*corr];
}


int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  
  MPI_Status *status;
  
  size_t notsend = 0;

  //10000 steps
  size_t steps = 1000;
  size_t part = 10;
  
  float p, E_buf, E, errorvalue, meanvalue;
  float T_buf, T = (world_size-(world_rank+1)) * 0.2;
  std::vector<float> energies;
  std::vector<float> uncorr;
  //
  //gpu setup
  //
  
  for(size_t j = 0; j < part; ++j)
  {
    for(size_t i = 0; i < steps; ++i)
    {
      //E = ising_MC...
      if(j > 100) //thermalize
      {
        //async copy energy to host
        energies.push_back(E);
       
        //calculate corr
        //calculate mean
        //calculate err
	uncorr = energies;
	removeCorr(uncorr);
	meanvalue = mean(uncorr);
	errorvalue = error(uncorr);
	uncorr.resize(0);
      }
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
  
    if(world_rank%2 == 0)
    {
      if(world_rank+1 != world_size)
      {
        MPI_Sendrecv(&E,     1, MPI_FLOAT, world_rank+1, 0,
    	             &E_buf, 1, MPI_FLOAT, world_rank,   0,
	             MPI_COMM_WORLD, status);
        MPI_Sendrecv(&T,     1, MPI_FLOAT, world_rank+1, 0,
	             &T_buf, 1, MPI_FLOAT, world_rank,   0,
	             MPI_COMM_WORLD, status);
      }
      
      if(world_rank != 0)
      {
        //p = random...
        if(p < exp((E-E_buf)/(T-T_buf))) //k_B = 1
        {
          MPI_Sendrecv(&T, 1, MPI_FLOAT, world_rank-1, 0,
	               &T, 1, MPI_FLOAT, world_rank,   0,
	               MPI_COMM_WORLD, status);
          T = T_buf;
          notsend = 0;
        }
        else
        {
         notsend++;
        }

	if(notsend == 10)
        {
	  std::cout << T << " " << meanvalue << " " << errorvalue << "\n";
          T += 0.05;
          notsend = 0;
        }
      }
    }
    else
    {
      if(world_rank+1 != world_size)
      {
        MPI_Sendrecv(&E,     1, MPI_FLOAT, world_rank+1, 0,
    	             &E_buf, 1, MPI_FLOAT, world_rank,   0,
	             MPI_COMM_WORLD, status);
        MPI_Sendrecv(&T,     1, MPI_FLOAT, world_rank+1, 0,
	             &T_buf, 1, MPI_FLOAT, world_rank,   0,
	             MPI_COMM_WORLD, status);
      }
      
      if(world_rank != 0)
      {
        //p = random...
        if(p < exp(fabs(E-E_buf)/fabs(T-T_buf))) //k_B = 1
        {
          MPI_Sendrecv(&T, 1, MPI_FLOAT, world_rank-1, 0,
	               &T, 1, MPI_FLOAT, world_rank,   0,
	               MPI_COMM_WORLD, status);
          T = T_buf;
          notsend = 0;
        }
        else
        {
         notsend++;
        }
      }

      if(notsend == 10)
      {
	std::cout << T << " " << meanvalue << " " << errorvalue << "\n";
        T += 0.05;
        notsend = 0;
      }
    }

    //
    //free gpu memory
    //
    
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::cout << T << " " << meanvalue << " " << errorvalue << "\n";

  MPI_Finalize();
  
  return 0;
}
