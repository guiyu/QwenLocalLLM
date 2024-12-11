cmake_minimum_required(VERSION 3.4.1)

# 添加GGML库
add_library(ggml STATIC
    ggml/ggml.c
)

# 设置GGML编译选项
target_compile_options(ggml PRIVATE
    -DGGML_USE_ACCELERATE=OFF
    -DGGML_USE_OPENBLAS=OFF
    -DGGML_USE_CUBLAS=OFF
    -DGGML_USE_CLBLAST=OFF
)

# 添加phi2库
add_library(phi2 SHARED
    phi2.cpp
)

# 链接库
target_link_libraries(phi2
    ggml
    android
    log
)

# 包含目录
target_include_directories(phi2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml
)