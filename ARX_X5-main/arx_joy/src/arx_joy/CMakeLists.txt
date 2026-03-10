cmake_minimum_required(VERSION 3.8)
project(arx_joy)

find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
        lib/arx_joy/include
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_joy)

add_executable(arx_joy src/arx_joy.cpp)
target_link_libraries(arx_joy libarx_joy.so)
ament_target_dependencies(arx_joy rclcpp std_msgs)

install(TARGETS
        arx_joy
        DESTINATION lib/${PROJECT_NAME})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/arx_joy/libarx_joy.so
        DESTINATION lib/
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()