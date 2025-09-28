cmake_minimum_required(VERSION 3.16)
project(eldercare_tracking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio tracking video dnn)

set(PIPELINE_SOURCES
    src/TrackingPipeline.cpp
    src/RescueStrategy.cpp
    src/CsrtTracker.cpp
    src/Quality.cpp
    src/BoxKalmanFilter.cpp
    src/NCNNSsdDetector.cpp
    src/StaticBoxDetector.cpp
    src/TemplateMatcherDetector.cpp
    src/OnnxSsdDetector.cpp)

add_library(eldercare_pipeline STATIC ${PIPELINE_SOURCES})

target_include_directories(eldercare_pipeline PUBLIC include)

target_link_libraries(eldercare_pipeline PUBLIC ${OpenCV_LIBS})

add_executable(eldercare_tracking
    src/main.cpp)

target_link_libraries(eldercare_tracking PRIVATE eldercare_pipeline)

add_executable(eldercare_tracking_otb
    src/otb_evaluator.cpp)

target_link_libraries(eldercare_tracking_otb PRIVATE eldercare_pipeline)
