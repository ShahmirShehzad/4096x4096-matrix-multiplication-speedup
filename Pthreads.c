#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MAX_THREADS 8

// Structure to hold thread arguments
typedef struct {
    int id;             // Thread ID
    int **matrix1;      // Matrix 1
    int **matrix2;      // Matrix 2
    int **result;       // Resultant Matrix
    int rows1, cols1;   // Dimensions of Matrix 1
    int rows2, cols2;   // Dimensions of Matrix 2
} ThreadArgs;

// Function to perform matrix multiplication for a portion of the resultant matrix
void *multiply(void *arg) {
    ThreadArgs *args = (ThreadArgs *)arg;

    // Calculate range for this thread
    int start = args->id * args->rows1 / MAX_THREADS;
    int end = (args->id == MAX_THREADS - 1) ? args->rows1 : (args->id + 1) * args->rows1 / MAX_THREADS;

    // Perform matrix multiplication for the assigned portion
    for (int i = start; i < end; i++) {
        for (int j = 0; j < args->cols2; j++) {
            for (int k = 0; k < args->cols1; k++) {
                args->result[i][j] += args->matrix1[i][k] * args->matrix2[k][j];
            }
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s input_file.txt output_file.txt\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];

    FILE *file = fopen(input_file, "r");
    if (file == NULL) {
        printf("Error: Could not open input file\n");
        return 1;
    }

    int matrix_data_type;
    int dimensions1[2], dimensions2[2];

    // Read matrix data from the input file
    fscanf(file, "%*d\n"); // Skip opcode
    fscanf(file, "%d\n", &matrix_data_type);
    fscanf(file, "%d %d\n", &dimensions1[0], &dimensions1[1]);
    int **matrix1 = (int **)malloc(dimensions1[0] * sizeof(int *));
    for (int i = 0; i < dimensions1[0]; i++) {
        matrix1[i] = (int *)malloc(dimensions1[1] * sizeof(int));
        for (int j = 0; j < dimensions1[1]; j++) {
            fscanf(file, "%d", &matrix1[i][j]);
        }
    }

    fscanf(file, "%d %d\n", &dimensions2[0], &dimensions2[1]);
    int **matrix2 = (int **)malloc(dimensions2[0] * sizeof(int *));
    for (int i = 0; i < dimensions2[0]; i++) {
        matrix2[i] = (int *)malloc(dimensions2[1] * sizeof(int));
        for (int j = 0; j < dimensions2[1]; j++) {
            fscanf(file, "%d", &matrix2[i][j]);
        }
    }

    fclose(file);

    // Perform matrix multiplication
    int **result = (int **)malloc(dimensions1[0] * sizeof(int *));
    for (int i = 0; i < dimensions1[0]; i++) {
        result[i] = (int *)calloc(dimensions2[1], sizeof(int));
    }

    pthread_t threads[MAX_THREADS];
    ThreadArgs args[MAX_THREADS];

    // Initialize thread arguments and create threads
    for (int i = 0; i < MAX_THREADS; i++) {
        args[i].id = i;
        args[i].matrix1 = matrix1;
        args[i].matrix2 = matrix2;
        args[i].result = result;
        args[i].rows1 = dimensions1[0];
        args[i].cols1 = dimensions1[1];
        args[i].rows2 = dimensions2[0];
        args[i].cols2 = dimensions2[1];
        pthread_create(&threads[i], NULL, multiply, (void *)&args[i]);
    }

    // Measure execution time
    clock_t start_time = clock();

    // Wait for all threads to finish
    for (int i = 0; i < MAX_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // Measure execution time
    clock_t end_time = clock();
    double execution_time = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Matrix multiplication executed in %f seconds\n", execution_time);

    // Write result to the output file
    FILE *output = fopen(output_file, "w");
    if (output == NULL) {
        printf("Error: Could not open output file\n");
        return 1;
    }

    fprintf(output, "Matrix data type: %d\n", matrix_data_type);
    fprintf(output, "Dimensions of the resultant matrix: %d x %d\n", dimensions1[0], dimensions2[1]);
    fprintf(output, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < dimensions1[0]; i++) {
        for (int j = 0; j < dimensions2[1]; j++) {
            fprintf(output, "%d ", result[i][j]);
        }
        fprintf(output, "\n");
    }

    fclose(output);

    // Free allocated memory
    for (int i = 0; i < dimensions1[0]; i++) {
        free(matrix1[i]);
    }
    free(matrix1);

    for (int i = 0; i < dimensions2[0]; i++) {
        free(matrix2[i]);
    }
    free(matrix2);

    for (int i = 0; i < dimensions1[0]; i++) {
        free(result[i]);
    }
    free(result);

    return 0;
}
