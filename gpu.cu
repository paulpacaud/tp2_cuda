/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2022                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "main.h"
#include "gpu.h"


/*-------------------------------------------------------------------------------*/
/* GPU symbols and global vars                                                   */
/*-------------------------------------------------------------------------------*/

// Symbols used by all kernels (Matrices on GPU devices) ------------------------
__device__ T_real GPU_A[SIZE][SIZE];
__device__ T_real GPU_B[SIZE][SIZE];
__device__ T_real GPU_C[SIZE][SIZE];

// Symbol and vars to call Cublas lib. ------------------------------------------
__device__ T_real GPU_Ctmp[SIZE][SIZE];   // New matrix buffer

T_real *AdrGPU_A = NULL;                  // Adresses of the symbols
T_real *AdrGPU_B = NULL;
T_real *AdrGPU_C = NULL;
T_real *AdrGPU_Ctmp = NULL; 

cublasHandle_t cublasHandle;              // Handle on the Cublas lib.


/*-------------------------------------------------------------------------------*/
/* Init and finalize the GPU device.                                             */
/*-------------------------------------------------------------------------------*/
void gpuInit(void)
{
  // Init of the GPU device -----------------------------------------------------
  cuInit(0);
  
  // Turn CPU arrays A, B and C into "locked" memory areas to speedup transfers--
  CHECK_CUDA_SUCCESS(cudaHostRegister(A,SIZE*SIZE*sizeof(T_real),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the A CPU array");
  CHECK_CUDA_SUCCESS(cudaHostRegister(B,SIZE*SIZE*sizeof(T_real),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the B CPU array");
  CHECK_CUDA_SUCCESS(cudaHostRegister(C,SIZE*SIZE*sizeof(T_real),
                                      cudaHostRegisterPortable),
                     "Turning into pinned memory the C CPU array");
  
  // Initializations to call Cublas lib. ----------------------------------------
  
  // - Extract address of GPU matrix "symbols" - useful when calling cuBLAS
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_A,GPU_A),"GPU_A adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_B,GPU_B),"GPU_B adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_C,GPU_C),"GPU_C adr extraction");
  CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **)&AdrGPU_Ctmp,GPU_Ctmp),"GPU_Ctmp adr extraction");
  
  // - Initialize CUBLAS lib usage
  CHECK_CUBLAS_SUCCESS(cublasCreate(&cublasHandle), "Init of the CUBLAS lib handle"); 
}


void gpuFinalize(void)
{
  // Turn "pinned (or locked)" CPU arrays into std array ------------------------
  CHECK_CUDA_SUCCESS(cudaHostUnregister(A),
                     "Turning into std memory the A CPU array");
  CHECK_CUDA_SUCCESS(cudaHostUnregister(B),
                     "Turning into std memory the B CPU array");
  CHECK_CUDA_SUCCESS(cudaHostUnregister(C),
                     "Turning into std memory the C CPU array");

  // Free CUBLAS lib usage ------------------------------------------------------
  CHECK_CUBLAS_SUCCESS(cublasDestroy(cublasHandle), "Free the CUBLAS lib");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                   */
/*-------------------------------------------------------------------------------*/
void gpuSetDataOnGPU(void)
{
  // Set GPU_A symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_A, &A, sizeof(A), 0, cudaMemcpyHostToDevice),
                    "Transfer A-->GPU_A");

  // Set GPU_B symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_B, &B, sizeof(B), 0, cudaMemcpyHostToDevice),
                    "Transfer B-->GPU_B");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                        */
/*-------------------------------------------------------------------------------*/
void gpuGetResultOnCPU(void)
{
  // Get GPU_C symbol
  CHECK_CUDA_SUCCESS(cudaMemcpyFromSymbol(&C, GPU_C, sizeof(C), 0, cudaMemcpyDeviceToHost),
                    "Transfer C<--GPU_C");
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU - 1D & generic matrix size              */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v0(void)
{
  // Index computations
  int row = blockIdx.y;
  int col = blockIdx.x*BLOCK_SIZE_X_K0 + threadIdx.x;
  T_real res = 0.0;

  // Matrix product computation
  for (int k = 0; k < SIZE; k++) {
    res += GPU_A[row][k] * GPU_B[k][col];
  }
  GPU_C[row][col] = res;
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU - 2D & generic matrix size              */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v1(void)
{
  // Index computations
  int row = blockIdx.y*BLOCK_SIZE_Y_K1 + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE_X_K1 + threadIdx.x;
  T_real res = 0.0;

  // Matrix product computation
  for (int k = 0; k < SIZE; k++) {
    res += GPU_A[row][k] * GPU_B[k][col];
  }
  GPU_C[row][col] = res;
}

/*-------------------------------------------------------------------------------*/
/* Kernel K2                                                                     */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v2(void)
{
  int NbStep = gridDim.x;

  int row = blockIdx.y*BLOCK_SIZE_XY_K2 + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE_XY_K2 + threadIdx.x;

  //Declaration of variables in Shared Memory
  __shared__ T_real shA[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ T_real shB[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ T_real shC[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  memset(shC, 0, sizeof(T_real)*BLOCK_SIZE_XY_K2*BLOCK_SIZE_XY_K2);

  //Computation loop (with synchronizations)
  for (int k = 0; k < NbStep; k++) {
    //Caching of data
    shA[threadIdx.y][threadIdx.x] = GPU_A[row][k*BLOCK_SIZE_XY_K2 + threadIdx.x];
    shB[threadIdx.y][threadIdx.x] = GPU_B[k*BLOCK_SIZE_XY_K2 + threadIdx.y][col];

    //Synchronization
    __syncthreads();

    //Computation of shared C
    for (int l=0; l< BLOCK_SIZE_XY_K2; l++) {
      shC[threadIdx.y][threadIdx.x] = shA[threadIdx.y][l] * shB[l][threadIdx.x];
    }

    __syncthreads();
  }

  //Storing results in global memory
  GPU_C[row][col] = shC[threadIdx.y][threadIdx.x];
}
  // successive threads in a warp access successive memory locations, it is coalesced

  // each thread makes NbStep*(1 access to GPU_A + 1 access to GPU_B) + 1 access to C = 2NbStep + 1
  // so for all threads (N² threads), it gives N² * (2NbStep + 1) = N² * (2N/BSX) + 1 

  // TD5 = 2N^3/BSX, et TD4: (2N + 1).N² et TD5 = TD4/BSXY
  // NbStep lecture rateau de A/NbStep lecture rateau de B 

  // TD5 = (2N + 1)/BSXY

  // Ws = Warp Size
  // en TD4, on avait N²/Ws * (2N + 1)
  // en TD5, on a réussi à diviser par BSXY

/*-------------------------------------------------------------------------------*/
/* Kernel K3                                                                     */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v3(void)
{
  //Declaration of variables in Shared Memory
  __shared__ float shA[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ float shB[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ float shC[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];

  // Index computations
  int row = blockIdx.y*BLOCK_SIZE_XY_K2 + threadIdx.y;
  int col = blockIdx.x*BLOCK_SIZE_XY_K2 + threadIdx.x;
  int NbStep = SIZE/BLOCK_SIZE_XY_K2;

  //Computation loop (with synchronizations)
  for (int k = 0; k < NbStep; k++) {
    //Caching of data
    if (row < SIZE && k*BLOCK_SIZE_XY_K2 + threadIdx.x < SIZE)
      shA[threadIdx.y][threadIdx.x] = GPU_A[row][k*BLOCK_SIZE_XY_K2 + threadIdx.x]; // !
    else
      shA[threadIdx.y][threadIdx.x] = 0;
    if (col < SIZE && k*BLOCK_SIZE_XY_K2 + threadIdx.y < SIZE)
      shB[threadIdx.y][threadIdx.x] = GPU_B[k*BLOCK_SIZE_XY_K2 + threadIdx.y][col]; // !
    else
      shB[threadIdx.y][threadIdx.x] = 0;
    //Synchronization
    __syncthreads();
    if (row < SIZE && col < SIZE) {
        //Computation of shared C
        for (int l=0; l< BLOCK_SIZE_XY_K2; l++) { // !
          shC[threadIdx.y][threadIdx.x] = shA[threadIdx.y][l] * shB[l][threadIdx.x];
        }
    }



    __syncthreads();
  }

  if (row < SIZE && col < SIZE) {
    //Storing results in global memory
    GPU_C[row][col] = shC[threadIdx.y][threadIdx.x]; // !
  }
}

/*-------------------------------------------------------------------------------*/
/* Transposition kernel using global memory and registers (slow version)         */
/*-------------------------------------------------------------------------------*/
__global__ void TransposeKernel_v0(T_real *MT, T_real *M, int mRow, int nCol)
{
 int row = threadIdx.y + blockIdx.y*BLOCK_SIZE_XY_KT0;
 int col = threadIdx.x + blockIdx.x*BLOCK_SIZE_XY_KT0;
 
 if (row < mRow && col < nCol)
   MT[col*mRow + row] = M[row*nCol + col];
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU.                                        */
/*-------------------------------------------------------------------------------*/
void gpuProduct(gkid_t kid)
{
 dim3 Dg = {0,0,0};   // Grid descriptor
 dim3 Db = {0,0,0};   // Block descriptor
 
 //T_real alpha;      // When using CUBLAS
 //T_real beta;       // When using CUBLAS

 switch(kid) {

 case GK0 : // Kernel v0 - 1D kernel using only resgisters and cache with generic matrix size
   // - init the grid of blocs
   Db.x = BLOCK_SIZE_X_K0;
   Db.y = 1;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_X_K0;
   Dg.y = SIZE;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v0<<<Dg,Db>>>();
   break;

 case GK1 : // kernel v1 : 2D kernel using only registers and cache with generic matrix size
   // - init the grid of blocs
   Db.x = BLOCK_SIZE_X_K1;
   Db.y = BLOCK_SIZE_Y_K1;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_X_K1;
   Dg.y = SIZE/BLOCK_SIZE_Y_K1;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v1<<<Dg,Db>>>();
   break;

 case GK2 : // kernel v2 : 2D kernel using the shared memories
   // - init the grid of blocs
   Db.x = BLOCK_SIZE_XY_K2;
   Db.y = BLOCK_SIZE_XY_K2;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_XY_K2;
   Dg.y = SIZE/BLOCK_SIZE_XY_K2;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v2<<<Dg,Db>>>();
   break;
  
 case GK3 : // kernel v3 : 2D kernel using the shared memories with generic matrix size
   break;

 case GK4 : // calling cublas gemm & user-defined transpose kernel
   break;
   
 case GK5 : // Calling cublas gemm & cublas geam kernels
   break;

 case GK6 : // Calling cublas gemm, using matrix math properties
   break;

 case GK7 : // Calling cublas gemmEx with std 32F datatypes, using Tensor cores
   break;

 case GK8 : // Calling cublas gemmEx with low precision on 32 bits datatypes, using Tensor cores
   break;

 default :
   fprintf(stderr,"Unknown GPU kernel!");
   exit(EXIT_FAILURE);
 } // End of switch
}




