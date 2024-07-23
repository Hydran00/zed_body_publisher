cmake_minimum_required(VERSION 3.8)
project(body_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXXFLAGS "-std=c++14")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

option(LINK_SHARED_ZED "Link with the ZED SDK shared executable" ON)
option(ENABLE_GUI "Enable OpenGL/ OpenCV display" ON)

if (NOT LINK_SHARED_ZED AND MSVC)
    message(FATAL_ERROR "LINK_SHARED_ZED OFF : ZED SDK static libraries not available on Windows")
endif()

find_package(ZED 3 REQUIRED)
find_package(CUDA REQUIRED)
if (ENABLE_GUI)
    add_definitions(-DENABLE_GUI)
    find_package(GLUT REQUIRED)
    find_package(GLEW REQUIRED)
    SET(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)
	find_package(OpenCV REQUIRED)
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${COMMON_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

IF(NOT WIN32)
    SET(SPECIAL_OS_LIBS "pthread")
ENDIF()

if (ENABLE_GUI)
    include_directories(${GLEW_INCLUDE_DIRS})
    include_directories(${GLUT_INCLUDE_DIR})
	include_directories(${OpenCV_INCLUDE_DIRS})
    
    link_directories(${GLEW_LIBRARY_DIRS})
    link_directories(${GLUT_LIBRARY_DIRS})
    link_directories(${OpenGL_LIBRARY_DIRS})
	link_directories(${OpenCV_LIBRARY_DIRS})

    SET(GUI_LIB ${OPENGL_LIBRARIES} ${OpenCV_LIBRARIES} ${GLUT_glut_LIBRARY} ${GLEW_LIBRARIES})
endif()

FILE(GLOB_RECURSE SRC_FILES src/*.c*)
FILE(GLOB_RECURSE HDR_FILES include/*.h*)

add_executable(${PROJECT_NAME} ${HDR_FILES} ${SRC_FILES})

ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs)
if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()


IF(WIN32)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS} ${UTILS_LIB} ${SPECIAL_OS_LIBS} ${GUI_LIB} ${OpenCV_LIBRARIES})
ELSE(WIN32)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${ZED_LIBS}  ${UTILS_LIB} ${SPECIAL_OS_LIBS} ${GUI_LIB} ${OpenCV_LIBRARIES})
ENDIF()

INSTALL (TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
