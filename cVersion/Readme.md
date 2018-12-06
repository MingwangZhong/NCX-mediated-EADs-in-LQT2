This is the c language version of the code. The speed is around 20 times slower as the CUDA version.


Compile:
```
gcc cell.c -o cell -lm
```

Run:

Stabilized RyR
	
	```./cell   0   .txt   10.0   400   4  0.04   7.14286   0.64  2.5   6   0```
	
Hyperactive RyR
	
	```
	./cell   0   .txt   10.0   400   4  2.52   3.125   0.64  2.5   6   0
	```
