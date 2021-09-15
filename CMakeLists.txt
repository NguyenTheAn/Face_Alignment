cmake_minimum_required(VERSION 2.8)

project(FaceStandarizer)

add_definitions(-std=c++11)
add_definitions("-Wall")

set(OpenCV_DIR "/home/ligirk/Workplace/facial_alignment/opencv")

include_directories(${OpenCV_DIR}/include/)

link_directories(${OpenCV_DIR}/lib/)

set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio
                opencv_face
                opencv_objdetect)
set(SRC
        FaceStandarizer.h
        FaceStandarizer.cpp
        main.cpp
        )

add_executable(FaceStandarizer ${SRC})
target_link_libraries(FaceStandarizer ${OpenCV_LIBS})

# file(GLOB FACESTANDARIZER_SRC FaceStandarizer.h
#         FaceStandarizer.cpp)
# set(FACESTANDARIZER_COMPILE_CODE ${FACESTANDARIZER_SRC})

# add_library( # Sets the name of the library.
#         FaceStandarizer

#         # Sets the library as a shared library.
#         STATIC

#         # Provides a relative path to your source file(s).
#         ${FACESTANDARIZER_COMPILE_CODE})


# install(TARGETS FaceStandarizer DESTINATION FaceStandarizer)
