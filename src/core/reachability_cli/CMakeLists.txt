cmake_minimum_required(VERSION 3.8)
project(reachability_cli)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_kinematics REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(random_numbers REQUIRED)
find_package(urdf REQUIRED)
find_package(srdfdom REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP)
find_package(HDF5 REQUIRED COMPONENTS C)

add_library(reachability_core
  src/model_loader.cpp
  src/npy_writer.cpp
)

target_include_directories(reachability_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(reachability_core
  moveit_core
  rclcpp
  random_numbers
  urdf
  srdfdom
  yaml-cpp
)

add_executable(fk_self_collision_cli src/fk_self_collision_cli.cpp)
target_link_libraries(fk_self_collision_cli reachability_core)
ament_target_dependencies(fk_self_collision_cli
  moveit_core
  rclcpp
  geometry_msgs
)

add_executable(ik_self_collision_cli src/ik_self_collision_cli.cpp)
target_link_libraries(ik_self_collision_cli reachability_core)
ament_target_dependencies(ik_self_collision_cli
  moveit_core
  moveit_kinematics
  moveit_msgs
  rclcpp
  geometry_msgs
)
target_link_libraries(ik_self_collision_cli
  moveit_kinematics::moveit_kdl_kinematics_plugin
)

add_executable(dataset_generator_cli src/dataset_generator_cli.cpp)
target_link_libraries(dataset_generator_cli reachability_core yaml-cpp)
ament_target_dependencies(dataset_generator_cli
  moveit_core
  moveit_kinematics
  moveit_msgs
  random_numbers
  rclcpp
  geometry_msgs
  yaml-cpp
)
target_link_libraries(dataset_generator_cli
  moveit_kinematics::moveit_kdl_kinematics_plugin
)
if(TARGET HDF5::HDF5)
  target_link_libraries(dataset_generator_cli HDF5::HDF5)
elseif(TARGET HDF5::C)
  target_link_libraries(dataset_generator_cli HDF5::C)
else()
  target_include_directories(dataset_generator_cli PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(dataset_generator_cli ${HDF5_LIBRARIES})
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(dataset_generator_cli OpenMP::OpenMP_CXX)
endif()

add_executable(orientation_dataset_cli src/orientation_dataset_cli.cpp)
target_link_libraries(orientation_dataset_cli reachability_core yaml-cpp)
ament_target_dependencies(orientation_dataset_cli
  moveit_core
  moveit_kinematics
  moveit_msgs
  random_numbers
  rclcpp
  geometry_msgs
  yaml-cpp
)
target_link_libraries(orientation_dataset_cli
  moveit_kinematics::moveit_kdl_kinematics_plugin
)
if(TARGET HDF5::HDF5)
  target_link_libraries(orientation_dataset_cli HDF5::HDF5)
elseif(TARGET HDF5::C)
  target_link_libraries(orientation_dataset_cli HDF5::C)
else()
  target_include_directories(orientation_dataset_cli PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(orientation_dataset_cli ${HDF5_LIBRARIES})
endif()
if(OpenMP_CXX_FOUND)
  target_link_libraries(orientation_dataset_cli OpenMP::OpenMP_CXX)
endif()

install(TARGETS
  reachability_core
  fk_self_collision_cli
  ik_self_collision_cli
  dataset_generator_cli
  orientation_dataset_cli
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
