cmake_minimum_required(VERSION 3.16)

project(c2d VERSION 1.0.0 DESCRIPTION "Library for the ACN C++ Environment" LANGUAGES CXX)
include(GNUInstallDirs)
add_library(c2d SHARED
                src/c2d.cpp
                src/atarienv.cpp
                src/replaybuffer.cpp)
set_target_properties(c2d PROPERTIES 
                    VERSION ${PROJECT_VERSION} 
                    INTERPROCEDURAL_OPTIMIZATION True)
target_include_directories(c2d PUBLIC src)

include(${PROJECT_SOURCE_DIR}/c2d/settings.cmake)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_config)
configure_file(cmake_config/c2d.pc.in c2d.pc @ONLY)
configure_file(cmake_config/common.hpp.in ${PROJECT_SOURCE_DIR}/src/common.hpp)
configure_file(cmake_config/hyperparameters.py.in ${PROJECT_SOURCE_DIR}/c2d/configured/hyperparameters.py)
configure_file(cmake_config/atarienv.py.in ${PROJECT_SOURCE_DIR}/c2d/configured/atarienv.py)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(OpenCV REQUIRED imgproc)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)
find_package(LZ4 REQUIRED)
find_package(ale REQUIRED)

set(cflags -march=native -Wall -Wextra -save-temps=obj)

target_compile_features(c2d PRIVATE cxx_std_20)
target_compile_options(c2d PRIVATE ${cflags})
target_link_libraries(c2d PRIVATE 
                        ${OpenCV_LIBS} 
                        ZLIB::ZLIB
                        Threads::Threads
                        LZ4::LZ4
                        ale::ale-lib)

install(TARGETS c2d
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_BINARY_DIR}/c2d.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
