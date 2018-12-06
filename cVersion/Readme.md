This is the c language version of the code to produce Fig. 3 A&D in Zhong et al<sup>1</sup>. 
The speed is around 20 times slower than the CUDA version.


## Compile
```
gcc cell.c -o cell -lm
```

## Run

Stabilized RyR
	
```
./cell   0   .txt   10.0   400   4  0.04   7.14286   0.64  2.5   6   0
```
	
Hyperactive RyR
	
```
./cell   0   .txt   10.0   400   4  2.52   3.125   0.64  2.5   6   0
```

[1]. Zhong, Mingwang, Colin M. Rees, Dmitry Terentyev, Bum-Rak Choi, Gideon Koren, and Alain Karma. "NCX-mediated subcellular Ca2+ dynamics underlying early afterdepolarizations in LQT2 cardiomyocytes." *Biophysical journal* 115, no. 6 (2018): 1019-1032.
