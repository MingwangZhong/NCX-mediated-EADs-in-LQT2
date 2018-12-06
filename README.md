# NCX mediated EADs in LQT2 cardiomyocytes

This code is to investigate the role of NCX and RyR activity in EAD formation in LQT2 myocytes, described in Ref. 1. Specifically, this code could produce the exact result in Fig. 3 D.

1. Zhong, Mingwang, Colin M. Rees, Dmitry Terentyev, Bum-Rak Choi, Gideon Koren, and Alain Karma. "NCX-mediated subcellular Ca<sup>2+</sup> dynamics underlying early afterdepolarizations in LQT2 cardiomyocytes." *Biophysical journal* 115, no. 6 (2018): 1019-1032. 

This code is based on an earlier implementation of a similar model described in:

2. Restrepo, J. G., Weiss, J. N., & Karma, A. (2008). Calsequestrin-mediated mechanism for cellular calcium transient alternans. *Biophysical journal*, 95(8), 3767-3789.
3. Terentyev, Dmitry, Colin M. Rees, Weiyan Li, Leroy L. Cooper, Hitesh K. Jindal, Xuwen Peng, Yichun Lu et al. "Hyperphosphorylation of RyRs underlies triggered activity in transgenic rabbit model of LQT2 syndrome." *Circulation research* 115, no. 11 (2014): 919-928.

This software is free software, distributed under the 2-clause BSD license. A copy of the license is included in the LICENSE file. We cordially ask that any published work derived from this code, or utilizing it references the above-mentioned published works.

## Implementation

For a simulation with 15 beats and pacing cycle length 4 s, the running time is ~2 hours on GeForce GTX TITAN. Another c language version of this code is also provided under the folder `cVersion/`, which is around 20 times slower than the CUDA version.

**Compile the code**

```
nvcc cell.cu -O3 -lm -arch sm_30  -o cell -w
```

**Run the code**

For the stabilized RyRs
```
./cell   0   .txt   10.0   400   4  0.04   7.14286   0.64  2.5   6   0
```

For the hyperactive RyRs
```
./cell   0   .txt   10.0   400   4  2.52   3.125   0.64  2.5   6   0
```
