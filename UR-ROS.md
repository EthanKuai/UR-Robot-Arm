# ROS Drivers

**`ur_client_library`** & **`ur_robot_driver`**

- https://docs.universal-robots.com/Universal_Robots_ROS_Documentation/rolling/doc/ur_client_library/doc/index.html
- https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/index.html

> [!info]
> In the backend it comms via URScripts.

## Installation
> Assumes within ROS workspace (easiest)

```shell
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y ros-$ROS_DISTRO-ur-client-library   # RTDE, Dashboard Server, URScript
sudo apt-get install -y ros-$ROS_DISTRO-ur                  # ROS
```

# `ur_client_library`

## C++ Build

```bash
tee -a CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.11.0) # That's the minimum required version for FetchContent
project(minimal_example)

include(FetchContent)
FetchContent_Declare(
  ur_client_library
  GIT_REPOSITORY https://github.com/UniversalRobots/Universal_Robots_Client_Library.git
  GIT_TAG        master
)

# This will download the ur_client_library and replace the `find_package(ur_client_library)` call.
FetchContent_MakeAvailable(ur_client_library)

add_executable(db_client main.cpp)
target_link_libraries(db_client ur_client_library::urcl)
EOF

mkdir build && cd build
cmake ..
cmake --build .
```

# `ur_robot_driver`

## Setup
> https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html

```shell

```
