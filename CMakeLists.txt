cmake_minimum_required(VERSION 3.12)

set(Darknet_MAJOR_VERSION 0)
set(Darknet_MINOR_VERSION 2)
set(Darknet_PATCH_VERSION 5)
set(Darknet_TWEAK_VERSION 2)
set(Darknet_VERSION ${Darknet_MAJOR_VERSION}.${Darknet_MINOR_VERSION}.${Darknet_PATCH_VERSION}.${Darknet_TWEAK_VERSION})


option(ENABLE_OPENCV "Enable OpenCV integration" ON)
set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
project(Darknet )

enable_language(CXX)

set(CMAKE_CXX_STANDARD 14)

set(INSTALL_BIN_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where exe and dll will be installed")
set(INSTALL_LIB_DIR      "${CMAKE_CURRENT_LIST_DIR}" CACHE PATH "Path where lib will be installed")
set(INSTALL_INCLUDE_DIR  "include/darknet"           CACHE PATH "Path where headers will be installed")
set(INSTALL_CMAKE_DIR    "share/darknet"             CACHE PATH "Path where cmake configs will be installed")


include(CheckLanguage)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package (Eigen3 REQUIRED )

set(CMAKE_DEBUG_POSTFIX d)

find_package(OpenMP)

include_directories(${OpenCV_INCLUDE_DIRS})

foreach(p LIB BIN INCLUDE CMAKE)
    set(var INSTALL_${p}_DIR)
    if(NOT IS_ABSOLUTE "${${var}}")
        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
    endif()
endforeach()

file(GLOB headers "${CMAKE_CURRENT_LIST_DIR}/src/*.hpp")
list(APPEND headers ${CMAKE_CURRENT_LIST_DIR}/include/darknet.hpp)

file(GLOB sources "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")
list(APPEND sources ${CMAKE_CURRENT_LIST_DIR}/src/image_opencv.cpp)

list(REMOVE_ITEM sources ${CMAKE_CURRENT_LIST_DIR}/src/darknet.cpp)

add_executable(darknet ${CMAKE_CURRENT_LIST_DIR}/src/darknet.cpp ${sources} ${headers})
target_include_directories(darknet PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>)

if(OpenCV_FOUND)
    target_link_libraries(darknet PRIVATE ${OpenCV_LIBS})
endif()

if(OPENMP_FOUND)
    target_link_libraries(darknet PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(darknet PRIVATE Threads::Threads)
target_link_libraries (darknet PRIVATE Eigen3::Eigen)

