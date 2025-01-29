#include <stdio.h>
#include <stdlib.h>

// Function to perform matrix multiplication
void matrix_multiply(int **matrix1, int **matrix2, int **result, int rows1, int cols1, int rows2, int cols2) {
    if (cols1 != rows2) {
        printf("Error: Incompatible matrix dimensions for multiplication\n");
        exit(1);
    }

    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

// Function to allocate memory for a matrix
int **allocate_matrix(int rows, int cols) {
    int **matrix = (int **)malloc(rows * sizeof(int *));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int *)malloc(cols * sizeof(int));
    }
    return matrix;
}

// Function to free memory allocated for a matrix
void free_matrix(int **matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./matrix_multiplication_c input_file.txt\n");
        return 1;
    }

    char *input_file = argv[1];
    char *out_file = argv[2];
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
    int **matrix1 = allocate_matrix(dimensions1[0], dimensions1[1]);
    for (int i = 0; i < dimensions1[0]; i++) {
        for (int j = 0; j < dimensions1[1]; j++) {
            fscanf(file, "%d", &matrix1[i][j]);
        }
    }

    fscanf(file, "%d %d\n", &dimensions2[0], &dimensions2[1]);
    int **matrix2 = allocate_matrix(dimensions2[0], dimensions2[1]);
    for (int i = 0; i < dimensions2[0]; i++) {
        for (int j = 0; j < dimensions2[1]; j++) {
            fscanf(file, "%d", &matrix2[i][j]);
        }
    }

    fclose(file);

    // Perform matrix multiplication
    int **result = allocate_matrix(dimensions1[0], dimensions2[1]);
    for (int i = 0; i < dimensions1[0]; i++) {
        for (int j = 0; j < dimensions2[1]; j++) {
            result[i][j] = 0;
        }
    }
    matrix_multiply(matrix1, matrix2, result, dimensions1[0], dimensions1[1], dimensions2[0], dimensions2[1]);

    // Write result to the output file
    FILE *output_file = fopen(argv[2], "w");
    fprintf(output_file, "Matrix data type: %d\n", matrix_data_type);
    fprintf(output_file, "Dimensions of the resultant matrix: %d x %d\n", dimensions1[0], dimensions2[1]);
    fprintf(output_file, "Rows of the resultant matrix after multiplication:\n");
    for (int i = 0; i < dimensions1[0]; i++) {
        for (int j = 0; j < dimensions2[1]; j++) {
            fprintf(output_file, "%d ", result[i][j]);
        }
        fprintf(output_file, "\n");
    }

    fclose(output_file);

    // Free allocated memory
    free_matrix(matrix1, dimensions1[0]);
    free_matrix(matrix2, dimensions2[0]);
    free_matrix(result, dimensions1[0]);

    return 0;
}
