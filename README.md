# Description
A simple Reinforcement Learning MCTS with c++

# Fix Nvtool
## Windows
```powershell
�ڵ� 59 �� find_package(CUDAToolkit REQUIRED) ֮�������������ݣ�
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