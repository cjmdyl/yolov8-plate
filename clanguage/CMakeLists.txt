cmake_minimum_required(VERSION 3.10)
project(YOLOv8_Detector CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 库
find_package(OpenCV REQUIRED)

# 查找 ONNX Runtime 库
find_package(ONNXRuntime REQUIRED)

# 添加你的可执行文件
add_executable(my_detector main.cpp)

# 链接必要的库
target_link_libraries(my_detector PRIVATE ${OpenCV_LIBS} ONNXRuntime::onnxruntime)