# Troubleshoot

## *unsupported gcc version*

As time of writing, latest cuda (10.2) only supports gcc up to gcc-8. But as you
can see, I take pleasure in using c++17/c++20 features, meaning I'm using gcc 9.

If you set `CXX=g++-9 CC=gcc-9` you will get errors when running
`find_package(CUDA)`. A simple trick is to symlink a supported gcc version next
to nvcc! For example:

```bash
export VERSION=8
sudo ln -s `which gcc-${VERSION}` /usr/local/cuda/bin/gcc 
sudo ln -s `which g++-${VERSION}` /usr/local/cuda/bin/g++
```
