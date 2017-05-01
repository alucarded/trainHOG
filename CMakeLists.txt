# LSD Plus
project(TRAINHOG)
cmake_minimum_required(VERSION 3.0.2)

set(EXEC_NAME trainhog)

###############################################################################
# Gather information, perform checks, set defaults

# Guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "In-source builds not allowed.
        Please make a new directory (named build) and run CMake from there.
        You may need to remove CMakeCache.txt. ")
endif()

###############################################################################
# Set up options

#Set up default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release"
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(SVMLIGHT 1)
set(LIBSVM 2)
if(TRAINHOG_USEDSVM EQUAL ${SVMLIGHT})
    set(CMAKE_CXX_FLAGS "-DTRAINHOG_USEDSVM=1")
elseif(TRAINHOG_USEDSVM EQUAL ${LIBSVM})
    set(CMAKE_CXX_FLAGS "-DTRAINHOG_USEDSVM=2")
else()
    #message( FATAL_ERROR "You need to set TRAINHOG_USEDSVM CMake variable to 1 - SVMLIGHT or 2 - LIBSVM." )
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -w -DNDEBUG")


###############################################################################
# Find Libraries

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

###############################################################################
# Add components


################################################################################
# Print configuration variables
message(STATUS "===============================================================")
message(STATUS "================  Configuration Options  ======================")
message(STATUS "Build flags                                                    ")
string(TOUPPER "${CMAKE_BUILD_TYPE}" cmake_build_type_toupper)
if(NOT MSVC AND NOT XCODE_VERSION)
    message(STATUS "  Build type                     : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C compilation flags            : ${CMAKE_C_FLAGS}"
                                "${CMAKE_C_FLAGS_${cmake_build_type_toupper}}")
    message(STATUS "  C++ compilation flags          : ${CMAKE_CXX_FLAGS} "
                                "${CMAKE_CXX_FLAGS_${cmake_build_type_toupper}}")
endif()

message(STATUS "===============================================================")
################################################################################

include_directories( ${TRAINHOG_SOURCE_DIR}/src )
include_directories( ${TRAINHOG_SOURCE_DIR}/third_party )

if(TRAINHOG_USEDSVM EQUAL ${SVMLIGHT})
    set(SRCS
        third_party/svmlight/svm_learn.h
        third_party/svmlight/svm_learn.c
        third_party/svmlight/svm_hideo.h
        third_party/svmlight/svm_hideo.c
        third_party/svmlight/svm_common.h
        third_party/svmlight/svm_common.c
        )
elseif(TRAINHOG_USEDSVM EQUAL ${LIBSVM})
    set(SRCS
        third_party/libsvm/svm.h
        third_party/libsvm/svm.cpp
        )
else()
    set(SRCS
        third_party/opencvsvm/opencvsvm.h
        )
endif()

set(TRAINHOG_MAIN src/train.cpp)

message( ${SRCS} )
add_executable(${EXEC_NAME} ${TRAINHOG_MAIN} ${SRCS})

target_link_libraries(${EXEC_NAME} ${OpenCV_LIBS})
