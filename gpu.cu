#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define DIMENSION 4096
#define BLOCK_SIZE 16


// CUDA kernel for matrix multiplication (32-bit integer)
__global__ void matrixMul32(int *matrix1, int *matrix2, int *matrix3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < DIMENSION && col < DIMENSION) {
        for (int k = 0; k < DIMENSION; ++k) {
            sum += matrix1[row * DIMENSION + k] * matrix2[k * DIMENSION + col];
        }
        matrix3[row * DIMENSION + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (64-bit integer)
__global__ void matrixMul64(long long int *matrix1, long long int *matrix2, long long int *matrix3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
    long long int sum = 0;

    if (row < DIMENSION && col < DIMENSION) {
        for (int k = 0; k < DIMENSION; ++k) {
            sum += matrix1[row * DIMENSION + k] * matrix2[k * DIMENSION + col];
        }
        matrix3[row * DIMENSION + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (32-bit float)
__global__ void matrixMulFloat(float *matrix1, float *matrix2, float *matrix3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < DIMENSION && col < DIMENSION) {
        for (int k = 0; k < DIMENSION; ++k) {
            sum += matrix1[row * DIMENSION + k] * matrix2[k * DIMENSION + col];
        }
        matrix3[row * DIMENSION + col] = sum;
    }
}

// CUDA kernel for matrix multiplication (64-bit double)
__global__ void matrixMulDouble(double *matrix1, double *matrix2, double *matrix3) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    if (row < DIMENSION && col < DIMENSION) {
        for (int k = 0; k < DIMENSION; ++k) {
            sum += matrix1[row * DIMENSION + k] * matrix2[k * DIMENSION + col];
        }
        matrix3[row * DIMENSION + col] = sum;
    }
}

// Function to perform matrix multiplication on the GPU
void matrixMultiplyGPU32int(int *matrix1, int *matrix2, int *matrix3) {
    int *d_matrix1, *d_matrix2, *d_matrix3;

    // Allocate memory on the device
    cudaMalloc((int **)&d_matrix1, DIMENSION * DIMENSION * sizeof(int));
    cudaMalloc((int **)&d_matrix2, DIMENSION * DIMENSION * sizeof(int));
    cudaMalloc((int **)&d_matrix3, DIMENSION * DIMENSION * sizeof(int));

    // Copy input matrices from host to device
    cudaMemcpy(d_matrix1, matrix1, DIMENSION * DIMENSION * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, DIMENSION * DIMENSION * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE, (DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel based on data type
    matrixMul32<<<gridSize, blockSize>>>(d_matrix1, d_matrix2, d_matrix3);

    // Copy the result matrix from device to host
    cudaMemcpy(matrix3, d_matrix3, DIMENSION * DIMENSION * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
}

void matrixMultiplyGPU64int(long long int *matrix1, long long int *matrix2, long long int *matrix3) {
    long long int *d_matrix1, *d_matrix2, *d_matrix3;

    // Allocate memory on the device
    cudaMalloc((void **)&d_matrix1, DIMENSION * DIMENSION * sizeof(long long int));
    cudaMalloc((void **)&d_matrix2, DIMENSION * DIMENSION * sizeof(long long int));
    cudaMalloc((void **)&d_matrix3, DIMENSION * DIMENSION * sizeof(long long int));

    // Copy input matrices from host to device
    cudaMemcpy(d_matrix1, matrix1, DIMENSION * DIMENSION * sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, DIMENSION * DIMENSION * sizeof(long long int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE, (DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel based on data type
    matrixMul64<<<gridSize, blockSize>>>(d_matrix1, d_matrix2, d_matrix3);

    // Copy the result matrix from device to host
    cudaMemcpy(matrix3, d_matrix3, DIMENSION * DIMENSION * sizeof(long long int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
}

void matrixMultiplyGPUfloat(float *matrix1, float *matrix2, float *matrix3) {
    float *d_matrix1, *d_matrix2, *d_matrix3;

    // Allocate memory on the device
    cudaMalloc((void **)&d_matrix1, DIMENSION * DIMENSION * sizeof(float));
    cudaMalloc((void **)&d_matrix2, DIMENSION * DIMENSION * sizeof(float));
    cudaMalloc((void **)&d_matrix3, DIMENSION * DIMENSION * sizeof(float));

    // Copy input matrices from host to device
    cudaMemcpy(d_matrix1, matrix1, DIMENSION * DIMENSION * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, DIMENSION * DIMENSION * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE, (DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel based on data type
    matrixMulFloat<<<gridSize, blockSize>>>(d_matrix1, d_matrix2, d_matrix3);

    // Copy the result matrix from device to host
    cudaMemcpy(matrix3, d_matrix3, DIMENSION * DIMENSION * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
}

void matrixMultiplyGPUdouble(double *matrix1, double *matrix2, double *matrix3) {
    double *d_matrix1, *d_matrix2, *d_matrix3;

    // Allocate memory on the device
    cudaMalloc((void **)&d_matrix1, DIMENSION * DIMENSION * sizeof(double));
    cudaMalloc((void **)&d_matrix2, DIMENSION * DIMENSION * sizeof(double));
    cudaMalloc((void **)&d_matrix3, DIMENSION * DIMENSION * sizeof(double));

    // Copy input matrices from host to device
    cudaMemcpy(d_matrix1, matrix1, DIMENSION * DIMENSION * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, matrix2, DIMENSION * DIMENSION * sizeof(double), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE, (DIMENSION + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel based on data type
    matrixMulDouble<<<gridSize, blockSize>>>(d_matrix1, d_matrix2, d_matrix3);

    // Copy the result matrix from device to host
    cudaMemcpy(matrix3, d_matrix3, DIMENSION * DIMENSION * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_matrix3);
}

void Run_32BitInt(char* input_file, char* output_file)
{
    	FILE* fptr = fopen(input_file, "r");
    	
    	if (fptr == NULL)
    	{
    		printf("File could not be opened\n");
    		exit(0);
    	}
    	
    	int programtype = 0;
    	int datatype = 0;
    	int dimension1 = 0, dimension2 = 0;
    	char garbage;
    	
    	fscanf(fptr, "%d", &programtype);
    	fscanf(fptr, "%d", &datatype);
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d", dimension1);
    		printf("DImension 2: %d", dimension2);
    		exit(0);
    	}
    	
    	int i = 0, j = 0;
    	
    	
    	
    	static int matrix1[4096][4096];
    	static int matrix2[4096][4096];
    	static int matrix3[4096][4096];
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%d", &matrix1[i][j]);
    		}
    	}
    	
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d\n", dimension1);
    		printf("DImension 2: %d\n", dimension2);
    		exit(0);
    	}
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%d", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	clock_t seconds_start;
    	clock_t seconds_finish;
    	
    	int k = 0;
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	seconds_start = time(NULL);

    	printf("%d %d \n",matrix1[0][0],matrix1[3][0]);
    	matrixMultiplyGPU32int((int *)matrix1, (int *)matrix2, (int *)matrix3);
    	
    	seconds_finish = time(NULL);
    	
    	double cputime = ((double) (seconds_finish - seconds_start)) / CLOCKS_PER_SEC * 1000;
    	printf("Time taken for calculations = %f\n", cputime);
    	
    	//printf("%d %d %d", matrix3[1044][1044], matrix3[2067][2067], matrix3[4004][0] );
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "1\n4096X4096\n");
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fprintf(fptr, "%d ", matrix3[i][j]);
    		}
    		fprintf(fptr, " \n");
    	}
    	
    	fclose(fptr);
    	
}
void Run_64BitInt(char* input_file, char* output_file)
{
	FILE* fptr = fopen(input_file, "r");
    	
    	if (fptr == NULL)
    	{
    		printf("File could not be opened\n");
    		exit(0);
    	}
    	
    	int programtype = 0;
    	int datatype = 0;
    	int dimension1 = 0, dimension2 = 0;
    	char garbage;
    	
    	fscanf(fptr, "%d", &programtype);
    	fscanf(fptr, "%d", &datatype);
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096 )
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d", dimension1);
    		printf("DImension 2: %d", dimension2);
    		exit(0);
    	}
    	
    	int i = 0, j = 0;
    	
    	static long long int matrix1[4096][4096];
    	static long long int matrix2[4096][4096];
    	static long long int matrix3[4096][4096];
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lli", &matrix1[i][j]);
    		}
    	}
    	
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d\n", dimension1);
    		printf("DImension 2: %d\n", dimension2);
    		exit(0);
    	}
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lli", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	clock_t seconds_start;
    	clock_t seconds_finish;
    	
    	int k = 0;
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	seconds_start = time(NULL);

    	
    	matrixMultiplyGPU64int((long long int *)matrix1, (long long int *)matrix2, (long long int *)matrix3);
    	
    	seconds_finish = time(NULL);
    	
    	double cputime = ((double) (seconds_finish - seconds_start)) / CLOCKS_PER_SEC * 1000;
    	printf("Time taken for calculations = %f\n", cputime);
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "3\n4096X4096\n");
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fprintf(fptr, "%lli ", matrix3[i][j]);
    		}
    		fprintf(fptr, " \n");
    	}
    	
    	fclose(fptr);
}
void Run_32BitFloat(char* input_file, char* output_file)
{
	FILE* fptr = fopen(input_file, "r");
    	
    	if (fptr == NULL)
    	{
    		printf("File could not be opened\n");
    		exit(0);
    	}
    	
    	int programtype = 0;
    	int datatype = 0;
    	int dimension1 = 0, dimension2 = 0;
    	char garbage;
    	
    	fscanf(fptr, "%d", &programtype);
    	fscanf(fptr, "%d", &datatype);
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d", dimension1);
    		printf("DImension 2: %d", dimension2);
    		exit(0);
    	}
    	
    	int i = 0, j = 0;
    	
    	static float matrix1[4096][4096];
    	static float matrix2[4096][4096];
    	static float matrix3[4096][4096];
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%g", &matrix1[i][j]);
    		}
    	}
    	
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d\n", dimension1);
    		printf("DImension 2: %d\n", dimension2);
    		exit(0);
    	}
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%g", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	clock_t seconds_start;
    	clock_t seconds_finish;
    	
    	int k = 0;
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	seconds_start = time(NULL);

    	
    	matrixMultiplyGPUfloat((float *)matrix1, (float *)matrix2, (float *)matrix3);
    	
    	
    	seconds_finish = time(NULL);
    	
    	double cputime = ((double) (seconds_finish - seconds_start)) / CLOCKS_PER_SEC * 1000;
    	printf("Time taken for calculations = %f\n", cputime);
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "2\n4096X4096\n");
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fprintf(fptr, "%g ", matrix3[i][j]);
    		}
    		fprintf(fptr, " \n");
    	}
    	
    	fclose(fptr);
}
void Run_64BitDouble(char* input_file, char* output_file)
{
	FILE* fptr = fopen(input_file, "r");
    	
    	if (fptr == NULL)
    	{
    		printf("File could not be opened\n");
    		exit(0);
    	}
    	
    	int programtype = 0;
    	int datatype = 0;
    	int dimension1 = 0, dimension2 = 0;
    	char garbage;
    	
    	fscanf(fptr, "%d", &programtype);
    	fscanf(fptr, "%d", &datatype);
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d", dimension1);
    		printf("DImension 2: %d", dimension2);
    		exit(0);
    	}
    	
    	int i = 0, j = 0;
    	
    	static double matrix1[4096][4096];
    	static double matrix2[4096][4096];
    	static double matrix3[4096][4096];
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lf", &matrix1[i][j]);
    		}
    	}
    	
    	fscanf(fptr, "%d%c%d", &dimension1, &garbage, &dimension2);
    	
    	if(dimension1 != 4096 || dimension2 != 4096)
    	{
    		printf("Invalid Dimensions given for matrix 1\n");
    		printf("DImension 1: %d\n", dimension1);
    		printf("DImension 2: %d\n", dimension2);
    		exit(0);
    	}
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lf", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	clock_t seconds_start;
    	clock_t seconds_finish;
    	
    	int k = 0;
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	seconds_start = time(NULL);

    	
    	matrixMultiplyGPUdouble((double *)matrix1, (double *)matrix2, (double *)matrix3);
    	
    	
    	seconds_finish = time(NULL);
    	
    	double cputime = ((double) (seconds_finish - seconds_start)) / CLOCKS_PER_SEC * 1000;
    	printf("Time taken for calculations = %f\n", cputime);
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "4\n4096X4096\n");
    	
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fprintf(fptr, "%g ", matrix3[i][j]);
    		}
    		fprintf(fptr, " \n");
    	}
    	
    	fclose(fptr);
}

int main(int argc, char** argv)
{
	if(argc != 3)
    	{
        	printf("2 command line arguments are expected to run the code. Exiting the code.\n");
        	return -1;
    	}
    	
    	char* input_file = argv[1];
    	char* output_file = argv[2];
    	
    	FILE* fptr = fopen(input_file, "r");
    	
    	if (fptr == NULL)
    	{
    		printf("File could not be opened\n");
    		exit(0);
    	}
    	
    	int programtype = 0;
    	int datatype = 0;
    	
    	fscanf(fptr, "%i", &programtype);
    	fscanf(fptr, "%i", &datatype);
    	
    	printf("ProgramType: %i\n", programtype);
    	printf("Data Type: %i\n", datatype);
    	
    	if(datatype == 1)
    	{
    		Run_32BitInt(input_file, output_file);
    	}
    	else if (datatype == 2)
    	{
    		Run_32BitFloat(input_file, output_file);
	}
	else if (datatype == 3)
    	{
    		Run_64BitInt(input_file, output_file);
	}
	else if (datatype == 4)
    	{
    		Run_64BitDouble(input_file, output_file);
	}
	else
	{
		printf("Invalid data type OpCode, Halting execution\n");
		exit(0);
	}
    	
    	
    	
    	
    	    	
    	return 0;
    	
}
