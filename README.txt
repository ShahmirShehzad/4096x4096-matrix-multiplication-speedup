To create 4096x4096 input file
gcc inputmul -o c_input
./c_input 1 6 input 		// Command Line arguments must be in this format: <DataTypeOpCode> <ProgramTypeOpCode> <Filename>

C code:
gcc C.c -o c -mavx2 -fopenmp
./c input output

Pthread code:
gcc Pthread.c -o pt -mavx2 -fopenmp
./pt input output

Python code:
python3 Python.py

Java code:
javac Java.java


Part 1:
gcc openMP-avx2.c -o part1 -mavx2 -fopenmp
./part1 input output				// <filename_for_input> <filename_for_output>

Part 2:
gcc openMP-22-avx2.c -o part2 -mavx2 -fopenmp
./part2 input output

Part 3:
Run the code on collab and select tpu as run time type or create tpu instance on google cloud


Part 4:
nvcc gpu.cu -o gpu		//you will need to install CUDA first
./gpu input output

------
-----
cat output		//display file
head -3 output			// displays 3 lines