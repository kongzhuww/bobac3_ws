#!/usr/bin/env python
# coding=utf-8

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# ------ 在这里定义你所有的场馆和坐标 ------
# 这是一个 Python 字典，把名字和 (X, Y) 坐标绑在一起
LOCATIONS = {
    "起点": (0.0, 0.0),    # <--- 这里加上了回到 0,0 的坐标
    "广州馆": (2.6, 1.1),
    "吉林馆": (2.6, 2.2),
    "北京馆": (2.6, 0.1),
    "深圳馆": (1.2, 1.1),
    "上海馆": (1.2, 2.2),
}

def send_goal(place_name):
    # 根据你输入的名字，去字典里把坐标掏出来
    x, y = LOCATIONS[place_name]
    
    rospy.loginfo(f"正在准备前往【{place_name}】...")
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    # 设定 5 秒超时如果没连上就报错
    if not client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("连不上底层导航系统了！请检查那一长串roslaunch终端是否启动正常。")
        return

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # 填入提取出来的坐标
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0 

    rospy.loginfo(f"指令下达！小车开始前往【{place_name}】(X:{x}, Y:{y})")
    client.send_goal(goal)

    # 等待小车到达目的地
    client.wait_for_result()
    rospy.loginfo(f"报告老板：小车已成功到达【{place_name}】！\n")

if __name__ == '__main__':
    # 只需要初始化一次节点（非常重要，否则会报错）
    rospy.init_node('auto_nav_commander', anonymous=True)
    
    # 用一个死循环来打造“测试菜单”，让你不用反复重新跑脚本
    while not rospy.is_shutdown():
        print("="*60)
        print("欢迎使用 Bobac3 场馆导航系统！可以前往的场馆有：")
        print(" | ".join(LOCATIONS.keys()))
        print("（输入 '退出' 或按 Ctrl+C 可结束程序）")
        print("="*60)
        
        # 让终端停下来，等待你输入场馆名字
        target = input("请输入你想去的地点名称（例如: 起点）：")
        
        if target == '退出':
            print("退出系统，再见！")
            break
            
        if target in LOCATIONS:
            # 如果名字是对的，就调用前往函数
            send_goal(target)
        else:
            print(f"❌ 查不到【{target}】这个指令，检查是不是打错字了！\n")