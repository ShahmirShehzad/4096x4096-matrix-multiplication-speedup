import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class testJ {

    // Function to perform matrix multiplication
    public static int[][] matrixMultiply(int[][] matrix1, int[][] matrix2) {
        int rows1 = matrix1.length;
        int cols1 = matrix1[0].length;
        int cols2 = matrix2[0].length;

        if (cols1 != matrix2.length) {
            throw new IllegalArgumentException("Error: Incompatible matrix dimensions for multiplication");
        }

        int[][] result = new int[rows1][cols2];

        for (int i = 0; i < rows1; i++) {
            for (int j = 0; j < cols2; j++) {
                for (int k = 0; k < cols1; k++) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.out.println("Usage: java MatrixMultiplication input_file.txt output_file.txt");
            System.exit(1);
        }

        String inputFileName = args[0];
        String outputFileName = args[1];

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFileName));
             FileWriter writer = new FileWriter(outputFileName)) {

            reader.readLine(); // Skip the opcode
            int matrixDataType = Integer.parseInt(reader.readLine());
            String[] dimensions1 = reader.readLine().split(" ");
            int rows1 = Integer.parseInt(dimensions1[0]);
            int cols1 = Integer.parseInt(dimensions1[1]);
            int[][] matrix1 = new int[rows1][cols1];
            for (int i = 0; i < rows1; i++) {
                String[] row = reader.readLine().split(" ");
                for (int j = 0; j < cols1; j++) {
                    matrix1[i][j] = Integer.parseInt(row[j]);
                }
            }

            String[] dimensions2 = reader.readLine().split(" ");
            int rows2 = Integer.parseInt(dimensions2[0]);
            int cols2 = Integer.parseInt(dimensions2[1]);
            int[][] matrix2 = new int[rows2][cols2];
            for (int i = 0; i < rows2; i++) {
                String[] row = reader.readLine().split(" ");
                for (int j = 0; j < cols2; j++) {
                    matrix2[i][j] = Integer.parseInt(row[j]);
                }
            }

            // Measure execution time before matrix multiplication
            long startTime = System.nanoTime();

            int[][] result = matrixMultiply(matrix1, matrix2);

            // Measure execution time after matrix multiplication
            long endTime = System.nanoTime();
            double executionTimeInSeconds = (endTime - startTime) / 1e9;
            System.out.println("Matrix multiplication executed in " + executionTimeInSeconds + " seconds");

            // Write the result to the output file
            writer.write("Matrix data type: " + matrixDataType + "\n");
            writer.write("Dimensions of the resultant matrix: " + result.length + "x" + result[0].length + "\n");
            writer.write("Rows of the resultant matrix after multiplication:\n");
            for (int[] row : result) {
                for (int elem : row) {
                    writer.write(elem + " ");
                }
                writer.write("\n");
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
