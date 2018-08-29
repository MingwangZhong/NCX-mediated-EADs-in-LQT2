
Compile:
	nvcc cell.cu -O3 -lm -arch sm_30  -o cell -w

Run:

	Stabilized RyR
		./cell   0   .txt   10.0   400   4  0.04   7.14286   0.64  2.5   6   0

	Hyperactive RyR
		./cell   0   .txt   10.0   400   4  2.52   3.125   0.64  2.5   6   0
