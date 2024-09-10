cmake_minimum_required(VERSION 3.0.2)
project(vit_inference)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  std_msgs
  message_generation
  message_runtime
  sensor_msgs
  image_transport
)
find_package(OpenCV 4 REQUIRED)

add_message_files(
  FILES
  MaterialDetected.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  LIBRARIES vit_inference
  CATKIN_DEPENDS cv_bridge rospy std_msgs message_runtime message_generation image_transport sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

find_package(OpenCV 4 REQUIRED)

catkin_install_python(PROGRAMS 
  scripts/yolo_vit_mat_detector.py
  scripts/material_detector_service.py
  scripts/webcam_pub.py
  scripts/webcam_sub.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)