cmake_minimum_required(VERSION 3.20)
project(growing_nca)

# ===================
# External libraries
# ===================
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(dependencies/lodepng)

add_subdirectory(dependencies/tclap)

find_package(OpenGL OPTIONAL_COMPONENTS)

# =================
# Common libraries
# =================
add_library(nca_common OBJECT
        common/model.h
        common/model.cpp
        common/utils.h
        common/utils.cpp
)
target_include_directories(nca_common INTERFACE common)
target_link_libraries(nca_common PUBLIC lodepng ${TORCH_LIBRARIES} TCLAP)

# ===========
# GUI target
# ===========
if ( OpenGL_FOUND )
    add_custom_target(TARGET nca_gui)

    add_subdirectory(dependencies/glfw-3.3.9)
    include_directories(nca_gui dependencies/glfw-3.3.9/include)

    enable_language(CUDA)
    find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    add_subdirectory(dependencies/glad/)

    add_executable(nca_gui
            gui/gui.cpp # main
            gui/gl.cpp
            gui/gl.h
            gui/cuda_interop.cpp
            gui/cuda_interop.h
    )
    target_include_directories(nca_gui PRIVATE
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(nca_gui nca_common OpenGL glad glfw ${GLFW_LIBRARIES} ${CUDART_LIBRARY})
endif()

# ======================
# Model training target
# ======================
add_executable(nca_train
        train/train.cpp # main
        train/single_step_lr.cpp
        train/single_step_lr.h
        train/utils.cpp
        train/utils.h
)

target_link_libraries(nca_train nca_common)