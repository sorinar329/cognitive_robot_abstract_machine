cmake_minimum_required(VERSION 3.5)
project(windturbine_model)

find_package(ament_cmake REQUIRED)

install(DIRECTORY models urdf launch rviz
DESTINATION share/${PROJECT_NAME})

ament_package()
