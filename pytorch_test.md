# PyTorch Test

## Installation
1. Download the libtorch library from [here](https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorchOpenMP/libtorch-v2.0.0.zip).
2. `unzip libtorch-v2.0.0.zip `
3. `cmake . -DCMAKE_PREFIX_PATH=./libtorch -DCMAKE_BUILD_TYPE=Release`
4. `make`

## Benchmark Result: `2453` kmeans per second on M1 Macbook Pro (single thread)

## Problem: Clustering Result seems to be wrong
Here is a sample output of the `pytorch_test.cpp`
```
-0.101562 -0.078125 -0.0703125 -0.0546875 -0.046875 -0.03125 -0.0234375 -0.015625 -0.0078125 0.0078125 0.015625 0.03125 0.0390625 0.046875 0.0546875 0.0703125 16
-10 -10 -0.265625 -0.257812 -0.25 -0.234375 -0.226562 -0.203125 -0.179688 -0.15625 -0.132812 -0.109375 -0.0859375 -0.0625 -0.0390625 0 16
-10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -10 -0.1875 -0.109375 -0.0546875 16
```
There are two problems:
* The range of the data is from `-1.8262` to `0.9873`. Therefore, none of the centers are expected to be -10. However, the output gives many -10s.
* Even though I set `k=32`, I still only get 16 returns from the `kmeans` function.