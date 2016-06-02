cmake_minimum_required(VERSION 2.8.3)
project(Kinefly)

find_package(
    catkin 
    REQUIRED COMPONENTS 
    message_generation 
    std_msgs
    sensor_msgs
    geometry_msgs
    dynamic_reconfigure
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  MsgAnalogIn.msg
  MsgCommand.msg
  MsgDigitalIn.msg
  MsgFloat32List.msg
  MsgFlystate.msg
  MsgState.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  SrvFloat32List.srv
  SrvTipdata.srv
  SrvTrackerdata.srv
  SrvWingdata.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
  cfg/kinefly.cfg
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES Kinefly
  CATKIN_DEPENDS message_runtime
#  DEPENDS system_lib
)

install(PROGRAMS
  nodes/kinefly.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
  launch/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.launch"
  PATTERN ".svn" EXCLUDE
)

