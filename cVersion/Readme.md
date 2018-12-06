This c language version code is around 20 times slower than the CUDA version.

## Compile
```
gcc cell.c -o cell -lm
```

## Run

Stabilized RyRs
	
```
./cell   0   .txt   10.0   400   4  0.04   7.14286   0.64  2.5   6   0
```
	
Hyperactive RyRs
	
```
./cell   0   .txt   10.0   400   4  2.52   3.125   0.64  2.5   6   0
```
