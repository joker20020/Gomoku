# Fix Nvtool
```powershell
在第 59 行 find_package(CUDAToolkit REQUIRED) 之后添加两行内容：
add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
set_property(TARGET CUDA::nvToolsExt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
```