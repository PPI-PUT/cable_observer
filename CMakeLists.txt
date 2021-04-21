cmake_minimum_required(VERSION 3.0.2)
project(cable_observer)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
)

catkin_python_setup()

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cable_observer
   CATKIN_DEPENDS
)

include_directories(
# include
 ${catkin_INCLUDE_DIRS}
)


