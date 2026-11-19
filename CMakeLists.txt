cmake_minimum_required(VERSION 3.21)
project(FaceVision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

option(FACE_BUILD_TESTS "Build regression tests" OFF)
option(FACE_AUTO_DEPLOY "Copy Windows runtime dependencies after build" ON)

if(MSVC)
    add_compile_options(/utf-8 /W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(Qt6 6.2 REQUIRED COMPONENTS Core Gui Widgets Sql Concurrent)
find_package(OpenCV 4.5 REQUIRED COMPONENTS
    core imgproc imgcodecs videoio objdetect flann)


set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    mainwindow.ui
    facerecognizer.cpp
    facerecognizer.h
    facedatabase.cpp
    facedatabase.h
    personmanagerdialog.cpp
    personmanagerdialog.h
    facesamplestorage.cpp
    facesamplestorage.h
    recognitionlogstorage.cpp
    recognitionlogstorage.h
    apppaths.cpp
    apppaths.h
    facematcher.cpp
    facematcher.h
    facetracker.cpp
    facetracker.h
    livenessanalyzer.cpp
    livenessanalyzer.h
    imageutils.cpp
    imageutils.h
)

qt_add_executable(${PROJECT_NAME} WIN32 ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Sql
    Qt6::Concurrent
    ${OpenCV_LIBS}
)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "FaceVision"
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/models"
    COMMENT "Copying face recognition models"
)

# CMake 会根据实际链接目标自动收集 OpenCV 等运行库，避免硬编码版本号和路径。
if(WIN32 AND FACE_AUTO_DEPLOY)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${PROJECT_NAME}>
                $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMAND_EXPAND_LISTS
        COMMENT "Copying linked runtime DLLs"
    )

    # 官方 OpenCV Windows 包的 DLL 通常位于 OpenCV_DIR 的相邻 bin 目录。
    # 使用通配方式复制，兼容不同 OpenCV 版本和 Debug/Release 文件名。
    get_filename_component(OPENCV_CONFIG_PARENT "${OpenCV_DIR}" DIRECTORY)
    set(OPENCV_RUNTIME_CANDIDATES
        "${OPENCV_CONFIG_PARENT}/bin"
        "${OpenCV_DIR}/../bin"
        "${OpenCV_DIR}/../../bin")
    foreach(OPENCV_RUNTIME_DIRECTORY IN LISTS OPENCV_RUNTIME_CANDIDATES)
        get_filename_component(OPENCV_RUNTIME_DIRECTORY
            "${OPENCV_RUNTIME_DIRECTORY}" ABSOLUTE)
        if(EXISTS "${OPENCV_RUNTIME_DIRECTORY}")
            file(GLOB OPENCV_RUNTIME_DLLS
                "${OPENCV_RUNTIME_DIRECTORY}/opencv_*.dll"
                "${OPENCV_RUNTIME_DIRECTORY}/opencv*.dll")
            foreach(OPENCV_RUNTIME_DLL IN LISTS OPENCV_RUNTIME_DLLS)
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${OPENCV_RUNTIME_DLL}"
                            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                    COMMENT "Copying ${OPENCV_RUNTIME_DLL}")
            endforeach()
            if(OPENCV_RUNTIME_DLLS)
                break()
            endif()
        endif()
    endforeach()

    get_target_property(QT_QMAKE_EXECUTABLE Qt6::qmake IMPORTED_LOCATION)
    if(QT_QMAKE_EXECUTABLE)
        get_filename_component(QT_BIN_DIRECTORY "${QT_QMAKE_EXECUTABLE}" DIRECTORY)
        find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS "${QT_BIN_DIRECTORY}")
    endif()
    if(WINDEPLOYQT_EXECUTABLE)
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                    --no-translations
                    --compiler-runtime
                    "$<TARGET_FILE:${PROJECT_NAME}>"
            COMMENT "Deploying Qt plugins and runtime libraries"
        )
    else()
        message(WARNING "windeployqt was not found. Run it manually before copying the program to another computer.")
    endif()
endif()

include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY models
    DESTINATION ${CMAKE_INSTALL_BINDIR})

if(FACE_BUILD_TESTS)
    include(CTest)
    enable_testing()

    add_executable(face_database_test
        tests/face_database_test.cpp
        facedatabase.cpp facedatabase.h
        facesamplestorage.cpp facesamplestorage.h
        recognitionlogstorage.cpp recognitionlogstorage.h
        apppaths.cpp apppaths.h
        imageutils.cpp imageutils.h
    )
    target_link_libraries(face_database_test PRIVATE
        Qt6::Core Qt6::Sql ${OpenCV_LIBS})
    target_include_directories(face_database_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
    add_test(NAME database_crud COMMAND face_database_test)

    add_executable(face_matcher_test
        tests/face_matcher_test.cpp
        facematcher.cpp facematcher.h
        facedatabase.h)
    target_link_libraries(face_matcher_test PRIVATE
        Qt6::Core ${OpenCV_LIBS})
    target_include_directories(face_matcher_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
    add_test(NAME matcher_index COMMAND face_matcher_test)

    # 图片回归测试只有在测试素材存在时才生成，不再因缺少 images 目录导致配置失败。
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/images/single_face.png")
        set(FACE_TEST_IMAGE "${CMAKE_CURRENT_BINARY_DIR}/testdata/single_face.png")
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/images/single_face.png"
                       "${FACE_TEST_IMAGE}" COPYONLY)

        add_executable(face_detector_smoke_test
            tests/face_detector_smoke_test.cpp
            facerecognizer.cpp facerecognizer.h)
        target_link_libraries(face_detector_smoke_test PRIVATE ${OpenCV_LIBS})
        target_include_directories(face_detector_smoke_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

        add_executable(face_feature_smoke_test
            tests/face_feature_smoke_test.cpp
            facerecognizer.cpp facerecognizer.h)
        target_link_libraries(face_feature_smoke_test PRIVATE ${OpenCV_LIBS})
        target_include_directories(face_feature_smoke_test PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})

        add_test(NAME detector_single_face
            COMMAND face_detector_smoke_test
                "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2023mar.onnx"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/face_recognition_sface_2021dec.onnx"
                "${FACE_TEST_IMAGE}")
        add_test(NAME feature_single_face
            COMMAND face_feature_smoke_test
                "${CMAKE_CURRENT_SOURCE_DIR}/models/face_detection_yunet_2023mar.onnx"
                "${CMAKE_CURRENT_SOURCE_DIR}/models/face_recognition_sface_2021dec.onnx"
                "${FACE_TEST_IMAGE}")
    endif()
endif()
