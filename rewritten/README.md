This is the clean code. It produces very similar results as the original code.

**Compile the code**

For the stabilized RyRs
```bash
nvcc cell_stable.cu -O3 -lm -arch sm_30  -o cell -w
```

For the hyperactive RyRs
```bash
nvcc cell_hyper.cu -O3 -lm -arch sm_30  -o cell -w
```

**Run the code**
```bash
./cell 0
```

## Results
![Alt text](Figure.svg)
