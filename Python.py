#!/usr/bin/env python3
import sys
import time

# Function to perform matrix multiplication
def matrix_multiply(matrix1, matrix2):
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Error: Incompatible matrix dimensions for multiplication")

    result = [[0 for _ in range(cols2)] for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

# Function to convert matrix elements to the appropriate data type
def convert_matrix(matrix, data_type):
    if data_type == 1:  # 32 bit integers
        return [[int(elem) for elem in row] for row in matrix]
    elif data_type == 2:  # 32 bit floats
        return [[float(elem) for elem in row] for row in matrix]
    elif data_type == 3:  # 64 bit ints
        return [[int(elem) for elem in row] for row in matrix]
    elif data_type == 4:  # 64 bit doubles
        return [[float(elem) for elem in row] for row in matrix]

if __name__ == "__main__":

    print(len(sys.argv))
    print(sys.argv[0])
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    print("INPUT::: ",input_file)
    print("OUTPUT::: ",output_file)

    with open(input_file, 'r') as file:
        lines = file.readlines()
        matrix_data_type = int(lines[1])
        dimensions1 = tuple(map(int, lines[2].split()))
        matrix1 = [line.split() for line in lines[3:3+dimensions1[0]]]
        dimensions2 = tuple(map(int, lines[3+dimensions1[0]].split()))
        matrix2 = [line.split() for line in lines[4+dimensions1[0]:]]

    # Convert matrix elements to the appropriate data type
    matrix1 = convert_matrix(matrix1, matrix_data_type)
    matrix2 = convert_matrix(matrix2, matrix_data_type)

    try:
        # Measure execution time before matrix multiplication
        start_time = time.time()

        result = matrix_multiply(matrix1, matrix2)

        # Measure execution time after matrix multiplication
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Matrix multiplication executed in {execution_time} seconds")

        # Write the result to the output file
        with open(output_file, 'w') as out_file:
            out_file.write(f"Matrix data type: {matrix_data_type}\n")
            out_file.write(f"Dimensions of the resultant matrix: {len(result)}x{len(result[0])}\n")
            out_file.write("Rows of the resultant matrix after multiplication:\n")
            for row in result:
                out_file.write(' '.join(map(str, row)) + '\n')

    except ValueError as e:
        print(e)
