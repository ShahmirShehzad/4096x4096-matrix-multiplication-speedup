#include "stdio.h"
#include "stdlib.h"

//Command Line arguments must be in this format: <DataTypeOpCode> <ProgramTypeOpCode> <Filename>

void Create_32BitInt(char* filename, int programType);
void Create_64BitInt(char* filename, int programType);
void Create_32BitFloat(char* filename, int programType);
void Create_64BitDouble(char* filename, int programType);

int main(int argc, char** argv )
{
    if(argc != 4)
    {
        printf("3 command line arguments are expected to run the code. Exiting the code.\n");
        return -1;
    }
    
    int opcode = atoi(argv[1]);
    int programtype = atoi(argv[2]);
    char* filename = argv[3];

    if(opcode == 1)
    {
	printf("Creating input file for 32 bit integers\n");
	printf("Filename: %s\n", filename);
	printf("Program OpCode: %i\n", programtype);
	Create_32BitInt(filename, programtype);
	
    }
    else if(opcode == 2)
    {
	printf("Creating input file for 32 bit floats\n");
	printf("Filename: %s\n", filename);
	printf("Program OpCode: %i\n", programtype);
	Create_32BitFloat(filename, programtype);
    }
    else if(opcode == 3)
    {
	printf("Creating input file for 64 bit integers\n");
	printf("Filename: %s\n", filename);
	printf("Program OpCode: %i\n", programtype);
	Create_64BitInt(filename, programtype);
    }
    else if(opcode == 4)
    {
	printf("Creating input file for 64 bit doubles\n");
	printf("Filename: %s\n", filename);
	printf("Program OpCode: %i\n", programtype);
	Create_64BitDouble(filename, programtype);
    }
    else 
    {
        printf("Unexpected opcode was given, exiting the code\n");
        printf("Opcode: %s", argv[1]);
        return -1;
    }

    return 0;
}

void Create_32BitInt(char* filename, int programtype)
{
    FILE* fptr;
    
    fptr = fopen(filename, "w");
    
    if (fptr == NULL)
    {
    	printf("File could not be created\n");
    	exit(0);
    }
    
    //printing OpCode for program Type
    fprintf(fptr, "%i\n",programtype);
    
    //Printing OpCode for matrix data type
    fprintf(fptr, "1\n");
    
    //Dimensions of First Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    int i = 0, j = 0;
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    fprintf(fptr, "%d ", (rand() % 720));
    	}
    	fprintf(fptr, "\n");
    }
    
    
    //Dimensions of Second Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    fprintf(fptr, "%d ", (rand() % 720));
    	}
    	fprintf(fptr, "\n");
    }
    
    fclose(fptr);
}
void Create_64BitInt(char* filename, int programType)
{
    FILE* fptr;
    
    fptr = fopen(filename, "w");
    
    if (fptr == NULL)
    {
    	printf("File could not be created\n");
    	exit(0);
    }
    
    //printing OpCode for program Type
    fprintf(fptr, "%i\n",programType);
    
    //Printing OpCode for matrix data type
    fprintf(fptr, "3\n");
    
    //Dimensions of First Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    int i = 0, j = 0;
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    long long int num = rand();
    	    num = num << 32;
    	    num = num | (rand());
    	    
    	    fprintf(fptr, "%lli ", (num % 47453130));
    	}
    	fprintf(fptr, "\n");
    }
    
    
    //Dimensions of Second Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    long long int num = rand();
    	    num = num << 32;
    	    num = num | (rand());
    	    
    	    fprintf(fptr, "%lli ", (num % 47453130));
    	    
    	}
    	fprintf(fptr, "\n");
    }
    
    fclose(fptr);
}
void Create_32BitFloat(char* filename, int programType)
{
    FILE* fptr;
    
    fptr = fopen(filename, "w");
    
    if (fptr == NULL)
    {
    	printf("File could not be created\n");
    	exit(0);
    }
    
    //printing OpCode for program Type
    fprintf(fptr, "%i\n",programType);
    
    //Printing OpCode for matrix data type
    fprintf(fptr, "2\n");
    
    //Dimensions of First Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    int i = 0, j = 0;
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    fprintf(fptr, "%g ", ((float)rand() - 0.001));
    	}
    	fprintf(fptr, "\n");
    }
    
    
    //Dimensions of Second Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    fprintf(fptr, "%g ", ((float)rand() - 0.001));
    	}
    	fprintf(fptr, "\n");
    }
    
    fclose(fptr);
}
void Create_64BitDouble(char* filename, int programType)
{
    FILE* fptr;
    
    fptr = fopen(filename, "w");
    
    if (fptr == NULL)
    {
    	printf("File could not be created\n");
    	exit(0);
    }
    
    //printing OpCode for program Type
    fprintf(fptr, "%i\n",programType);
    
    //Printing OpCode for matrix data type
    fprintf(fptr, "4\n");
    
    //Dimensions of First Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    int i = 0, j = 0;
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    long long int num = rand();
    	    num = num << 32;
    	    num = num | (rand());
    	    
    	    fprintf(fptr, "%g ", ((double)num));
    	}
    	fprintf(fptr, "\n");
    }
    
    
    //Dimensions of Second Matrix(Currently hardcoding as 4096)
    fprintf(fptr, "4096X4096\n");
    
    
    for (i = 0; i < 4096; i++)
    {
    	for (j = 0; j < 4096; j++)
    	{
    	    long long int num = rand();
    	    num = num << 32;
    	    num = num | (rand());
    	    
    	    fprintf(fptr, "%g ", ((double)num));
    	    
    	}
    	fprintf(fptr, "\n");
    }
    
    fclose(fptr);
}
