cmake_minimum_required(VERSION 3.0.0)
project(dialogflow)

find_package(catkin REQUIRED COMPONENTS
  rospy std_msgs
  human_interactions sound_processing # My packages
)

catkin_package(
 CATKIN_DEPENDS rospy std_msgs
)
