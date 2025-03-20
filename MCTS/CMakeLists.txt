# CMakeList.txt: MCTS 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  MESSAGE(STATUS "On Linux Platform")
  set(CMAKE_PREFIX_PATH "/home/jdy/Software/Libtorch/libtorch/share/cmake/Torch")
  include_directories("/home/jdy/Software/Libtorch/libtorch/include")
  include_directories("/home/jdy/Software/Libtorch/libtorch/include/torch/csrc/api/include")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  MESSAGE(STATUS "On Windows Platform")
  set(CMAKE_PREFIX_PATH "D:/Libtorch/libtorch/share/cmake/Torch")
  include_directories("D:/Libtorch/libtorch/include")
  include_directories("D:/Libtorch/libtorch/include/torch/csrc/api/include")
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")

find_package(Torch REQUIRED)

# 将源代码添加到此项目的可执行文件。
include_directories("include")
set(SOURCES
	src/main.cpp
	src/game.cpp
	src/mcts.cpp
	src/gomoku.cpp
	src/model.cpp
	src/train.cpp
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (MCTS ${SOURCES})

target_link_libraries(MCTS "${TORCH_LIBRARIES}")


if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET MCTS PROPERTY CXX_STANDARD 20)
endif()

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET MCTS
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:MCTS>)
endif (MSVC)

# TODO: 如有需要，请添加测试并安装目标。
