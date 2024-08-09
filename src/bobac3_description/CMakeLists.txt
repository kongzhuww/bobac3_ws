cmake_minimum_required(VERSION 2.8.3)

project(bobac3_description)

catkin_package()

foreach(dir config launch meshes xacro)
  install(DIRECTORY ${dir}/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

