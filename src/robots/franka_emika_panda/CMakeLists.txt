cmake_minimum_required(VERSION 3.8)
project(franka_emika_panda)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
