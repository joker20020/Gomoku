# Fix Nvtool
```powershell
�ڵ� 59 �� find_package(CUDAToolkit REQUIRED) ֮������������ݣ�
add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
set_property(TARGET CUDA::nvToolsExt APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CUDAToolkit_INCLUDE_DIRS}")
```