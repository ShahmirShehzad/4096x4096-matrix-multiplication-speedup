#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "immintrin.h"
#include "omp.h"


struct arg_int32 {
	int (*matrix1)[4096];
	int (*matrix2)[4096];
	int (*matrix3)[4096];
	int start_row;
	int end_row;
};

struct arg_int64 {
	long long int (*matrix1)[4096];
	long long int (*matrix2)[4096];
	long long int (*matrix3)[4096];
	int start_row;
	int end_row;
};

struct arg_float {
	float (*matrix1)[4096];
	float (*matrix2)[4096];
	float (*matrix3)[4096];
	int start_row;
	int end_row;
};

struct arg_double {
	double (*matrix1)[4096];
	double (*matrix2)[4096];
	double (*matrix3)[4096];
	int start_row;
	int end_row;
};

void Run_32BitInt(char* input_file, char* output_file);
void Run_64BitInt(char* input_file, char* output_file);
void Run_32BitFloat(char* input_file, char* output_file);
void Run_64BitDouble(char* input_file, char* output_file);

void* Run_32BitInt_p(void* args)
{
	struct arg_int32 *arguments = args;
	
	int i = 0, j = 0, k = 0;
	
	#pragma omp parallel for private(i,j,k) shared(arguments)
    	for (i = arguments->start_row; i < arguments->end_row; i++) 
    	{
  		for (k = 0; k < 4096; k++)	
		{
			for (j = 0; j < 4096; j+=8) 
    			{
    				__m256i sum = _mm256_setzero_si256();
    			
      				__m256i bc_mat1 = _mm256_set1_epi32(arguments->matrix1[i][k]);
      				__m256i vec_mat2 = _mm256_loadu_si256((__m256i*)&arguments->matrix2[k][j]);
      				__m256i prod = _mm256_mullo_epi32(bc_mat1, vec_mat2);
      				
      				
      				__m256i vec_mat3 = _mm256_loadu_si256((__m256i*)&arguments->matrix3[i][j]);
      				sum = _mm256_add_epi32(vec_mat3, prod);
      				_mm256_storeu_si256((__m256i*)&arguments->matrix3[i][j], sum);
    			}
    		}
	}
	
	
	return NULL;
}
void* Run_64BitInt_p(void* args)
{
	struct arg_int64 *arguments = args;
	
	printf("%d,  %d\n", arguments->start_row, arguments->end_row);
	
	int i = 0, j = 0, k = 0;
    	
    	#pragma omp parallel for private(i,j,k) shared(arguments)
    	for (i = arguments->start_row; i < arguments->end_row; i++)
    	{
        	for (j = 0; j < 4096; j += 4)
        	{
            		__m256i sum = _mm256_setzero_si256();
            		for (k = 0; k < 4096; k++)
            		{
                		__m256i bc_mat1 = _mm256_set1_epi64x(arguments->matrix1[i][k]);
                		__m256i vec_mat2 = _mm256_loadu_si256((__m256i *)&arguments->matrix2[k][j]);
                		__m256i prod = _mm256_mul_epi32(bc_mat1, vec_mat2);
                		sum = _mm256_add_epi64(sum, prod);
            		}

            		_mm256_storeu_si256((__m256i *)&arguments->matrix3[i][j], sum);
        	}
    	}
	
	
	return NULL;
}
void* Run_32BitFloat_p(void* args)
{
	struct arg_float *arguments = args;
	
	printf("%d,  %d\n", arguments->start_row, arguments->end_row);
	
	int i = 0, j = 0, k = 0;
	
	
    	#pragma omp parallel for private(i,j,k) shared(arguments)
    	for (i = arguments->start_row; i < arguments->end_row; i++)
    	{
        	for (j = 0; j < 4096; j += 8)
        	{
            		__m256 sum = _mm256_setzero_ps();
            		for (k = 0; k < 4096; k++)
            		{
                		__m256 bc_mat1 = _mm256_set1_ps(arguments->matrix1[i][k]);
                		__m256 vec_mat2 = _mm256_loadu_ps(&arguments->matrix2[k][j]);
                		__m256 prod = _mm256_mul_ps(bc_mat1, vec_mat2);
                		sum = _mm256_add_ps(sum, prod);
            		}

            		_mm256_storeu_ps(&arguments->matrix3[i][j], sum);
        	}
    	}
	
	
	return NULL;
}

void* Run_64BitDouble_p(void* args)
{
	struct arg_double *arguments = args;
	
	printf("%d,  %d\n", arguments->start_row, arguments->end_row);
	
	int i = 0, j = 0, k = 0;
    	
    	#pragma omp parallel for private(i,j,k) shared(arguments)
    	for (i = arguments->start_row; i < arguments->end_row; i++)
    	{
        	for (j = 0; j < 4096; j += 4)
        	{
            		__m256d sum = _mm256_setzero_pd();
            		for (k = 0; k < 4096; k++)
            		{
                		__m256d bc_mat1 = _mm256_set1_pd(arguments->matrix1[i][k]);
                		__m256d vec_mat2 = _mm256_loadu_pd(&arguments->matrix2[k][j]);
                		__m256d prod = _mm256_mul_pd(bc_mat1, vec_mat2);
                		sum = _mm256_add_pd(sum, prod);
            		}

            		_mm256_storeu_pd(&arguments->matrix3[i][j], sum);
        	}
    	}
	
	
	return NULL;
}

int main(int argc, char** argv)
{
	#pragma omp parallel
	printf("%d threads\n",omp_get_num_threads());
	
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
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
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
    	
    	struct arg_int32 args;
    	
    	seconds_start = time(NULL);
    	
    	args.matrix1 = matrix1;
    	args.matrix2 = matrix2;
    	args.matrix3 = matrix3;
    	args.start_row = 0;
    	args.end_row = 4096;
    	
    	Run_32BitInt_p((void*)&args);
    	
    	seconds_finish = time(NULL);
    	double timetaken = ((double)(seconds_finish - seconds_start) * 1000.0 / CLOCKS_PER_SEC);
    	
    	printf("Time taken for calculations = %f\n", (timetaken));
    	
    	//printf("%d %d %d", matrix3[1044][1044], matrix3[2067][2067], matrix3[4004][0] );
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "1\n4096X4096\n");
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lli", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	time_t seconds_start;
    	time_t seconds_finish;
    	
    	int k = 0;
    	
    	printf("Running C matrix mul for 64 bit integers\n");
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	struct arg_int64 args;
    	
    	seconds_start = time(NULL);
    	
    	args.matrix1 = matrix1;
    	args.matrix2 = matrix2;
    	args.matrix3 = matrix3;
    	args.start_row = 0;
    	args.end_row = 4096;
    	
    	Run_64BitInt_p((void*)&args);
    	
    	seconds_finish = time(NULL);
    	
    	printf("Time taken for calculations = %ld\n", (seconds_finish - seconds_start) );
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "3\n4096X4096\n");
    	
    	#pragma parallel for private(i,j)
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
    	
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%g", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	time_t seconds_start;
    	time_t seconds_finish;
    	
    	int k = 0;
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	struct arg_float args;
    	
    	seconds_start = time(NULL);
    	
    	args.matrix1 = matrix1;
    	args.matrix2 = matrix2;
    	args.matrix3 = matrix3;
    	args.start_row = 0;
    	args.end_row = 4096;
    	
    	Run_32BitFloat_p((void*)&args);
    	
    	
    	seconds_finish = time(NULL);
    	
    	printf("Time taken for calculations = %ld\n", (seconds_finish - seconds_start) );
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "2\n4096X4096\n");
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
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
    	
    	#pragma parallel for private(i,j)
    	for (i = 0; i < 4096; i++)
    	{
    		for (j = 0; j < 4096; j++)
    		{
    			fscanf(fptr, "%lf", &matrix2[i][j]);
    		}
    	}
    	
    	fclose(fptr);
    	
    	time_t seconds_start;
    	time_t seconds_finish;
    	
    	int k = 0;
    	
    	printf("Running C matrix mul for 64 bit doubles\n");
    	
    	int task_size = 4096;
    	int task = 0;
    	
    	struct arg_double args;
    	
    	seconds_start = time(NULL);
    	
    	args.matrix1 = matrix1;
    	args.matrix2 = matrix2;
    	args.matrix3 = matrix3;
    	args.start_row = 0;
    	args.end_row = 4096;
    	
    	Run_64BitDouble_p((void*)&args);
    	
    	
    	seconds_finish = time(NULL);
    	
    	
    	printf("Time taken for calculations = %ld\n", (seconds_finish - seconds_start) );
    	
    	fptr = fopen(output_file, "w");
    	
    	fprintf(fptr, "4\n4096X4096\n");
    	
    	#pragma parallel for private(i,j)
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
