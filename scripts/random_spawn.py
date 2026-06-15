#!/usr/bin/env python3
"""
随机生成灭火器和火焰标记
每次巡逻前运行，随机决定哪个馆有火、哪个馆缺灭火器
约束：至少1个馆有火，至少1个馆缺灭火器
"""
import rospy
import random
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose, Point, Quaternion

# 5个馆的墙面位置 & 朝向
# 墙面都在东侧(x~2.97/1.47)，机器人从西往东看
PAVILIONS = {
    1: {"x": 2.967, "y": 0.2,  "name": "北京馆"},
    2: {"x": 2.967, "y": 1.1,  "name": "广州馆"},
    3: {"x": 2.967, "y": 2.2,  "name": "吉林馆"},
    4: {"x": 1.467, "y": 2.2,  "name": "上海馆"},
    5: {"x": 1.467, "y": 1.1,  "name": "深圳馆"},
}

# 读模型 SDF（用 model:// 路径不行，直接用文件路径）
import os
GAZEBO_MODEL_DIR = os.path.expanduser("~/.gazebo/models/rei_2025raicom")
FIRE_SDF_PATH = os.path.join(GAZEBO_MODEL_DIR, "fire_hazard", "model.sdf")
EXT_SDF_PATH  = os.path.join(GAZEBO_MODEL_DIR, "fire_extinguisher", "model.sdf")

def load_sdf(path):
    with open(path) as f:
        return f.read()


def spawn(name, sdf_str, x, y, z, roll=0, pitch=0, yaw=0):
    """在指定位置和姿态生成模型"""
    rospy.wait_for_service('/gazebo/spawn_sdf_model', timeout=10)
    try:
        spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        import math
        from geometry_msgs.msg import Quaternion as Q
        # 欧拉角 -> 四元数：roll=绕X, pitch=绕Y, yaw=绕Z
        cr, sr = math.cos(roll/2), math.sin(roll/2)
        cp, sp = math.cos(pitch/2), math.sin(pitch/2)
        cy, sy = math.cos(yaw/2), math.sin(yaw/2)
        q = Q()
        q.w = cr*cp*cy + sr*sp*sy
        q.x = sr*cp*cy - cr*sp*sy
        q.y = cr*sp*cy + sr*cp*sy
        q.z = cr*cp*sy - sr*sp*cy
        pose = Pose(Point(x, y, z), q)
        resp = spawn_srv(name, sdf_str, "", pose, "world")
        if resp.success:
            rospy.loginfo(f"  [+] {name}")
        else:
            rospy.logerr(f"  [-] {name}: {resp.status_message}")
        return resp.success
    except Exception as e:
        rospy.logerr(f"  [!] {name}: {e}")
        return False


def delete(name):
    try:
        rospy.wait_for_service('/gazebo/delete_model', timeout=5)
        srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        resp = srv(name)
        if resp.success:
            rospy.loginfo(f"  [x] {name}")
        return resp.success
    except:
        return False


def main():
    # node already initialized by caller; standalone call use rospy.init_node("random_spawn_markers")
    pass

    fire_sdf   = load_sdf(FIRE_SDF_PATH)
    ext_sdf    = load_sdf(EXT_SDF_PATH)

    # ---- 1. 删除上一轮 ----
    rospy.loginfo("=== Cleaning up ===")
    for i in range(1, 6):
        delete(f"fire_hazard_{i}")
        delete(f"fire_extinguisher_{i}")
    rospy.sleep(0.5)

    # ---- 2. 随机决定 ----
    points = [1, 2, 3, 4, 5]

    # 至少1个起火，最多2个
    fire_count = random.randint(1, 2)
    fire_pts = set(random.sample(points, fire_count))

    # 至少1个缺灭火器，最多2个
    noext_count = random.randint(1, 2)
    noext_pts = set(random.sample(points, noext_count))

    hasext_pts = set(points) - noext_pts

    rospy.loginfo(f"Fire at:      {sorted(fire_pts)}")
    rospy.loginfo(f"No ext at:    {sorted(noext_pts)}")
    rospy.loginfo(f"Has ext at:   {sorted(hasext_pts)}")

    # ---- 3. 生成 ----
    rospy.loginfo("=== Spawning ===")
    for idx, cfg in PAVILIONS.items():
        wx, wy = cfg["x"], cfg["y"]

        if idx in fire_pts:
            # 火焰：贴墙，右移
            spawn(f"fire_hazard_{idx}", fire_sdf,
                  wx - 0.005, wy + 0.15, 0.2, roll=1.57, yaw=4.7124)

        if idx in hasext_pts:
            # 灭火器：贴墙，右移
            spawn(f"fire_extinguisher_{idx}", ext_sdf,
                  wx - 0.005, wy + 0.15, 0.2, roll=1.57, yaw=4.7124)

    rospy.loginfo("=== Done ===")
    rospy.loginfo("Expected anomalies:")
    for i in fire_pts:
        rospy.loginfo(f"  - {PAVILIONS[i]['name']}: FIRE!")
    for i in noext_pts:
        rospy.loginfo(f"  - {PAVILIONS[i]['name']}: MISSING extinguisher!")


if __name__ == "__main__":
    main()
