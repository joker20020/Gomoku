# Description
A simple Reinforcement Learning MCTS with c++

# Usage
## clone the repo
```powershell
git clone https://github.com/joker20020/Gomoku.git
```

## download libtorch
download libtorch from [here](https://pytorch.org/)

## build
- modify `CMakeLists.txt` in `line7、9、9`(on linux) or `line12、13、14`(on windows)
- use cmake to build MakeFile
- make

## run
```powershell
./MCTS -m modelPath -d saveDirectory -n modelPoolNumber
```
- `-m` or `--model`: the path of the model to load
- `-d` or `--directory`: the directory to save the model
- `-n` or `--num`: the number of models in the model pool

# Fix Nvtool
## Windows
```powershell
#find line 
find_package(CUDAToolkit REQUIRED)
#add after
add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
set_property(TARGET CUDA::nvToolsExt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
```
## Linux
- Download [nvtx3](https://github.com/NVIDIA/NVTX/releases
- Change `/path/to/file/libtorch/share/cmake/Caffe2/public/cuda.cmake` line 172
```powershell
# nvToolsExt
if(USE_SYSTEM_NVTX)
  find_path(nvtx3_dir NAMES nvtx3)
else()
  find_path(nvtx3_dir NAMES nvtx3 PATHS "${PROJECT_SOURCE_DIR}/third_party/NVTX/c/include" NO_DEFAULT_PATH)
endif()
```
to
```powershell
if(USE_SYSTEM_NVTX)
  find_path(nvtx3_dir NAMES nvtx3 PATHS "/path/to/file/NVTX-3.1.1/c/include")
else()
  find_path(nvtx3_dir NAMES nvtx3 PATHS "/path/to/file/NVTX-3.1.1/c/include")
endif()
```