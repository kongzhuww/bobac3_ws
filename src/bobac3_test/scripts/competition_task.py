#!/usr/bin/env python3
import rospy
import actionlib
import subprocess
import json
import os
import time
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from face_rec.msg import face_results
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Twist
from ar_pose.srv import Track, TrackRequest
from relative_move.srv import SetRelativeMove, SetRelativeMoveRequest

# 入库方向说明：yaw=0 时摄像头朝外（朝向走道）
# TEB 已开启 allow_init_with_backwards_motion，规划器自动倒车入库停在此朝向
LOCATIONS = {
    "起点":  (0.05, -0.1, 0.0),
    "北京馆": (2.55, 0.1,  0.0),
    "广州馆": (2.55, 1.1,  0.0),
    "吉林馆": (2.55, 2.1,  0.0),
    "深圳馆": (1.1, 1.1,  0.0),
    "上海馆": (1.1, 2.1,  0.0),
}

# 任务二：巡检点（5个指定位置，顺序巡检，全部 yaw=0）
INSPECTION_POINTS = [
    ("巡检点1", 2.55, 0.1,  0.0),
    ("巡检点2", 2.55, 1.1,  0.0),
    ("巡检点3", 2.55, 2.1,  0.0),
    ("巡检点4", 1.1, 2.1,  0.0),
    ("巡检点5", 1.1, 1.1,  0.0),
]

# 充电桩位置（大体位置：x=0.8, y=1.94, yaw=-1.57 即 -90度朝下）
CHARGING_STATION = (0.45, 1.94, -1.57)          

# 巡检指令关键词
INSPECTION_COMMANDS = [
    "开始执行巡检任务", "开始巡检", "执行巡检", "启动巡检",
    "开始巡逻", "开始检查", "巡检", "巡逻",
]

# 切换到任务二的语音指令
TASK2_SWITCH_COMMANDS = [
    "切换到任务二", "切换到任务2", "切换任务二", "切换任务2",
    "任务二", "任务2", "管理模式", "进入任务二", "进入任务2",
]

# 切换到任务三的语音指令
TASK3_SWITCH_COMMANDS = [
    "切换到任务三", "切换到任务3", "切换任务三", "切换任务3",
    "任务三", "任务3", "训练模式", "进入任务三", "进入任务3",
    "开始训练", "训练",
]

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AUDIO_DIR = os.path.join(PKG_DIR, "audio")
MODEL_DIR = os.path.join(PKG_DIR, "models", "vosk-model-small-cn-0.22")
SAMPLE_RATE = 16000
LISTEN_TIMEOUT = 10
NAV_TIMEOUT = 90

def strip_guan(name):
    """去掉场馆名称末尾的'馆'字，匹配音频文件名"""
    return name.rstrip("馆")

def play_audio(filename):
    """播报语音"""
    path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(path):
        subprocess.run(['aplay', '-q', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    rospy.logwarn("Audio not found: " + filename)
    return False

def extract_place(text):
    """从语音指令中提取目标场馆"""
    if not text:
        return None

    # 完整匹配
    for venue in ["广州馆", "吉林馆", "北京馆", "深圳馆", "上海馆"]:
        if venue in text:
            return venue

    # 简称匹配
    short = {"广州":"广州馆", "吉林":"吉林馆", "北京":"北京馆", "深圳":"深圳馆", "上海":"上海馆"}
    for k, v in short.items():
        if k in text:
            return v

    return None

def extract_command(text):
    """从语音中检测是否包含巡检指令"""
    if not text:
        return False
    for cmd in INSPECTION_COMMANDS:
        if cmd in text:
            return True
    return False

def extract_charge_command(text):
    """从语音中检测是否包含直接充电指令"""
    if not text:
        return False
    return "充电" in text

def extract_switch_to_task2(text):
    """从语音中检测是否要切换到任务二"""
    if not text:
        return False
    for cmd in TASK2_SWITCH_COMMANDS:
        if cmd in text:
            return True
    return False

def extract_switch_to_task3(text):
    """从语音中检测是否要切换到任务三（训练模式）"""
    if not text:
        return False
    for cmd in TASK3_SWITCH_COMMANDS:
        if cmd in text:
            return True
    return False

def yaw_to_quaternion(yaw):
    """将 yaw 角 (弧度) 转为 geometry_msgs/Quaternion"""
    from geometry_msgs.msg import Quaternion
    return Quaternion(x=0, y=0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))

def send_nav_goal(place, timeout=NAV_TIMEOUT):
    """导航到目标位置，返回是否成功到达"""
    if place not in LOCATIONS:
        return False

    x, y, yaw = LOCATIONS[place]

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    if not client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("move_base server not available")
        return False

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)

    rospy.loginfo("Navigating to: %s (yaw=%.2f)" % (place, yaw))
    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(timeout))

    if finished and client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Successfully reached: " + place)
        return True
    return False

class SpeechRecognizer:
    def __init__(self, model_dir):
        import vosk
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_dir)
    
    def listen_once(self, timeout=LISTEN_TIMEOUT):
        import vosk, pyaudio

        rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)

        # 只在初始化麦克风时屏蔽 ALSA 噪音，初始化完就恢复
        saved_stderr = None
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved_stderr = os.dup(2)
            os.dup2(devnull, 2)
            os.close(devnull)
        except:
            pass

        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                            input=True, frames_per_buffer=4000)
            stream.start_stream()
        finally:
            # 恢复 stderr，让麦克风工作时的错误可见
            if saved_stderr is not None:
                os.dup2(saved_stderr, 2)
                os.close(saved_stderr)

        rospy.loginfo("[Mic] 麦克风已激活，开始监听...")

        start = time.time()
        result = ""

        while time.time() - start < timeout:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    result = res["text"]
                    break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if not result:
            result = json.loads(rec.FinalResult()).get("text", "")

        rospy.loginfo("[Mic] 识别结果: %s" % (result if result else "(空)"))
        return result

class CompetitionTask:
    def __init__(self, mode='visitor'):
        """
        mode: 'visitor' — 任务一：参观模式，人脸唤醒后语音导航
              'admin'   — 任务二：管理模式，识别人脸判断管理员
              'train'   — 任务三：训练模式，打字控制，跳过人脸
        """
        self.mode = mode
        self.face_flag = False
        self.recognized_name = ""   # 识别到的中文名
        self.is_admin = False       # 是否为管理员
        self.current_venue = None
        self.voice_cmd = None       # Windows 端发来的语音指令
        self.robot_x = 0.0          # 机器人当前位置
        self.robot_y = 0.0
        self.robot_orientation = None  # odom 四元数朝向
        self.at_origin = True       # 是否在原点

        rospy.Subscriber('/face_detection', face_results, self.face_cb)
        rospy.Subscriber('/voice_command', String, self.voice_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/yolo/detections', String, self._detection_cb)
        self._last_detections = ""

        # 通知 Windows 端开始/停止人脸检测和语音识别
        self.face_trigger = rospy.Publisher('/face_trigger', String, queue_size=1)
        self.voice_trigger_pub = rospy.Publisher('/voice_trigger', String, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        time.sleep(0.5)  # 等 publisher 注册完成

        rospy.loginfo("Mode: %s, ready" % mode)

    def odom_cb(self, msg):
        """跟踪机器人当前位置和朝向，判断是否在原点"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_orientation = msg.pose.pose.orientation
        dist = (self.robot_x**2 + self.robot_y**2)**0.5
        self.at_origin = dist < 0.5

    def face_cb(self, msg):
        """接收 Windows 端发来的人脸检测结果"""
        if self.face_flag:
            return  # 已经收到结果，忽略重复
        if len(msg.face_data) == 0:
            return
        self.face_flag = True
        if hasattr(msg, 'admin_name') and msg.admin_name:
            self.recognized_name = msg.admin_name
            self.is_admin = getattr(msg, 'is_admin', False)
            rospy.loginfo("Face: %s (admin=%s)" % (msg.admin_name, self.is_admin))
        else:
            self.recognized_name = ""
            self.is_admin = False
            rospy.loginfo("Face detected!")

    def voice_cb(self, msg):
        """接收 Windows 端发来的语音识别结果"""
        self.voice_cmd = msg.data
        rospy.loginfo("[Voice] 收到: %s" % msg.data)

    def wait_for_voice(self, prompt=""):
        """等待 Windows 端 /voice_command 语音指令（阻塞直到收到）"""
        rospy.loginfo("\n" + "="*30 + " >>> 请说话 (%s) <<< " % prompt + "="*30)

        # 通知 Windows 打开麦克风
        self.voice_trigger_pub.publish("start")
        
        self.voice_cmd = None

        while self.voice_cmd is None and not rospy.is_shutdown():
            time.sleep(0.1)

        if rospy.is_shutdown():
            return ""

        result = self.voice_cmd
        self.voice_cmd = None
        rospy.loginfo("[Voice] 结果: %s" % result)
        return result

    def run(self):
        rate = rospy.Rate(10)

        # 任务三（训练模式）跳过人脸检测，直接进
        if self.mode == 'train':
            self.do_training_task()
            return

        while not rospy.is_shutdown():
            # ===== 1. 等待回到原点 =====
            rospy.loginfo("[Run] 等待回到原点...")
            while not self.at_origin and not rospy.is_shutdown():
                rate.sleep()
            if rospy.is_shutdown():
                break

            # ===== 2. 触发 Windows 端人脸检测（优化防抖机制）=====
            rospy.loginfo("[Run] 已在原点，触发人脸检测 (%s 模式)" % self.mode)
            self.face_flag = False
            self.recognized_name = ""
            self.is_admin = False

            # 【核心修改点】：只发送一次触发信号，不再使用短周期的 while 轰炸
            self.face_trigger.publish(String(data=self.mode))
            rospy.loginfo("[Run] 已单次发送 /face_trigger: %s，等待 Windows 端响应..." % self.mode)
            
            # 设置最大等待时间为 15 秒，期间每 0.1 秒检查一次 Windows 的返回状态
            wait_start = time.time()
            while not self.face_flag and (time.time() - wait_start < 15.0) and not rospy.is_shutdown():
                time.sleep(0.1)
                
            if rospy.is_shutdown():
                break

            # 如果在 15 秒内没有任何人脸反馈，强行发送 stop 指令重置 Windows 并重新进入原点等待
            if not self.face_flag:
                rospy.logwarn("[Run] 触发人脸检测超时（未见人脸），发送 stop 信号清空 Windows 状态")
                self.face_trigger.publish(String(data='stop'))
                time.sleep(1.0)
                continue

            # 成功接收到人脸信号后，将标志位归零
            self.face_flag = False
            
            # 【重要安全切断】：立刻向 Windows 端补发一个 'stop' 信号，清除 ROS 队列积压并安抚 Windows 锁
            self.face_trigger.publish(String(data='stop'))
            rospy.loginfo("[Run] 成功获取人脸数据，已发送 'stop' 信号关闭 Windows 端人脸检测")

            # ===== 3. 任务一：访客模式（任意人脸）=====
            if self.mode == 'visitor':
                self.do_visitor_task()

                # 检查是否在任务一中切换到了任务二
                if self.mode == 'admin':
                    rospy.loginfo("[Run] Mode switched to admin, re-triggering face detection...")
                    continue
                elif self.mode == 'train':
                    rospy.loginfo("[Run] Mode switched to training, entering task 3...")
                    self.do_training_task()
                    # 训练模式结束后回到循环
                    self.mode = 'visitor'
                    continue    # 回到循环顶部，重新触发人脸检测（管理员识别）

            # ===== 4. 任务二：管理模式（需要管理员）=====
            elif self.mode == 'admin':
                if self.is_admin and self.recognized_name:
                    self.do_admin_task()
                else:
                    rospy.logwarn("[Run] 非注册管理员，拒绝执行任务，1秒后重新进入等待...")
                    time.sleep(1.0)

                # 检查是否切换到了训练模式
                if self.mode == 'train':
                    rospy.loginfo("[Run] Admin switched to training, entering task 3...")
                    self.do_training_task()
                    self.mode = 'visitor'
                    continue

            rate.sleep()

    # ============================================================
    # 任务一：迎宾导览模式
    #   评分：人脸唤醒30 + 语音交互10 + 导览导航40 + 返回出发区30 = 110分
    # ============================================================
    def do_visitor_task(self):
        # ===== 阶段1：人脸检测唤醒（30分）=====
        # 播报："你好，欢迎您的到来！有什么需要帮助的吗？"
        rospy.loginfo("[Task1] Face detected, activating...")
        play_audio("welcome.wav")
        time.sleep(1.0)

        # ===== 阶段2：语音交互应答（10分）=====
        # 正确识别"可以带我去参观XX馆吗？"等自然语音指令
        venue = None
        for attempt in range(3):
            text = self.wait_for_voice(prompt="attempt %d/3" % (attempt+1))

            rospy.loginfo("[Task1] Heard: %s" % text)

            # 检测是否要切换到任务二
            if extract_switch_to_task2(text):
                rospy.loginfo("[Task1] Switch to task 2 requested!")
                play_audio("switch_task2.wav")
                self.mode = 'admin'
                return

            # 检测是否要切换到任务三
            if extract_switch_to_task3(text):
                rospy.loginfo("[Task1] Switch to task 3 (training) requested!")
                self.mode = 'train'
                return

            venue = extract_place(text)

            if venue:
                rospy.loginfo("[Task1] Venue recognized: " + venue)
                self.current_venue = venue
                # 确认播报："好的，带您去XX馆"
                play_audio("confirm_" + strip_guan(venue) + ".wav")
                break
            else:
                rospy.logwarn("[Task1] No venue in: %s" % text)
                if attempt < 2:
                    # 没听清，提示再说一次
                    play_audio("retry_command.wav")
                    time.sleep(0.3)

        if not venue:
            rospy.logwarn("[Task1] Failed to recognize venue after 3 attempts")
            return

        # ===== 阶段3：迎宾引导导航（40分）=====
        # 起点直行入馆，馆间先转身再倒车入库（摄像头保持朝外）
        nav_success = self.navigate_to_venue(venue)
        if not nav_success:
            rospy.logwarn("[Task1] Navigation to %s failed" % venue)
            self.navigate_to_origin()
            return

        # 到达后播放场馆介绍
        rospy.loginfo("[Task1] Reached %s, playing introduction..." % venue)
        play_audio("intro_" + strip_guan(venue) + ".wav")
        time.sleep(1.5)

        # ===== 阶段4：返回出发区（30分）=====
        # 播报归位语音并导航返回起点（导航10 + 未压线10 + 语音10）
        back_file = "back_" + venue + ".wav"
        if not os.path.exists(os.path.join(AUDIO_DIR, back_file)):
            back_file = "back.wav"                 # fallback 通用返程语音
        rospy.loginfo("[Task1] Returning: " + back_file)
        play_audio(back_file)
        time.sleep(0.5)

        rospy.loginfo("[Task1] Navigating back to start...")
        self.navigate_to_origin()

        # 重置状态
        self.recognized_name = ""
        self.is_admin = False
        self.current_venue = None
        rospy.loginfo("[Task1] Mission complete!")

    # ============================================================
    # 任务二：管理模式
    #   人脸识别管理员 → 随机生成标记 → 语音指令 → 多点巡检 → 异常检测 → 充电
    # ============================================================
    def do_admin_task(self):
        # -------- 阶段1：管理员识别 --------
        if self.is_admin and self.recognized_name:
            rospy.loginfo("[Task2] Admin recognized: %s" % self.recognized_name)

            # 播报："欢迎管理员XXX进入系统"
            admin_file = "admin_%s.wav" % self.recognized_name
            admin_path = os.path.join(AUDIO_DIR, admin_file)
            if os.path.exists(admin_path):
                play_audio(admin_file)
            else:
                rospy.logwarn("[Task2] No audio for admin: %s" % self.recognized_name)
        else:
            rospy.loginfo("[Task2] Unknown person, not admin")
            play_audio("not_admin.wav")
            self.recognized_name = ""
            self.is_admin = False
            return

        # -------- 阶段1.5：随机生成灭火器和火焰标记 --------
        # rospy.loginfo("[Task2] Spawning random fire/extinguisher markers...")
        # scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts")
        # scripts_dir = os.path.abspath(scripts_dir)
        # import sys
        # if scripts_dir not in sys.path:
        #     sys.path.insert(0, scripts_dir)
        # import random_spawn
        # random_spawn.main()          # 官方环境已由 ./reinovo_bobac3_sim 自动生成，此处注释避免重复


        # -------- 阶段2：等待巡检指令 --------
        rospy.loginfo("[Task2] Waiting for inspection command...")
        play_audio("wait_command.wav")             # "请下达巡检指令"

        cmd_received = False
        skip_inspection = False
        for attempt in range(3):
            text = self.wait_for_voice(prompt="巡检指令 %d/3" % (attempt+1))

            # 检测是否要切换到任务三
            if extract_switch_to_task3(text):
                rospy.loginfo("[Task2] Switch to task 3 (training) requested!")
                self.mode = 'train'
                return

            if extract_charge_command(text):
                cmd_received = True
                skip_inspection = True
                rospy.loginfo("[Task2] Charge command received: " + text)
                # 直接播报准备充电的语音
                play_audio("low_battery.wav") 
                break

            if extract_command(text):
                cmd_received = True
                skip_inspection = False
                rospy.loginfo("[Task2] Inspection command received: " + text)
                play_audio("inspection_start.wav") # "收到指令，开始执行巡检任务"
                break
            else:
                rospy.loginfo("[Task2] Retry %d: %s" % (attempt+1, text))
                if attempt < 2:
                    play_audio("retry_command.wav")  # "请再说一次"

        if not cmd_received:
            rospy.logwarn("[Task2] No inspection command received")
            play_audio("inspection_cancel.wav")    # "未收到指令，取消巡检"
            self.recognized_name = ""
            self.is_admin = False
            return

        if not skip_inspection:
            # -------- 阶段3：多点自主巡检 --------
            anomalies = self.do_inspection()
    
            # -------- 阶段4：巡检完成，播报结果 --------
            rospy.loginfo("[Task2] Inspection complete, anomalies: %d" % len(anomalies))
            if len(anomalies) > 0:
                play_audio("inspection_anomaly_summary.wav")  # TODO: 待生成
                for a in anomalies:
                    rospy.logwarn("  Anomaly: %s at %s" % (a[0], a[1]))
            else:
                play_audio("inspection_complete.wav")  # "巡检任务完成，一切正常"

        # -------- 阶段5：自主对接充电 --------
        self.do_charging()

        # -------- 阶段6：返回起点 --------
        rospy.loginfo("[Task2] Charging complete, returning to origin...")
        self.navigate_to_origin()

        # 重置状态
        self.recognized_name = ""
        self.is_admin = False
        rospy.loginfo("[Task2] Mission complete!")

    # ============================================================
    # 多点巡检（任务二阶段3）
    #   依次导航到5个巡检点，每个点检测异常
    # ============================================================
    def do_inspection(self):
        anomalies = []

        for i, (name, x, y, yaw) in enumerate(INSPECTION_POINTS):
            rospy.loginfo("[Task2] Inspecting: %s (%.1f, %.1f, yaw=%.2f)" % (name, x, y, yaw))

            # 使用倒车入库策略导航到巡检点
            success = self._navigate_with_reverse(x, y, yaw)

            if not success:
                rospy.logwarn("[Task2] Failed to reach %s, skipping" % name)
                continue

            rospy.loginfo("[Task2] Reached %s" % name)
            play_audio("point_%s.wav" % name)

            # AI视觉异常检测
            fire = self.detect_fire()
            extinguisher = self.detect_extinguisher()

            if fire or not extinguisher:
                # 只要存在任何隐患，先统一播放 4 秒警报音频（仅一次）
                play_audio("fire_hazard.wav")
                
            # 建立巡检点与语音前缀的映射
            pavilion_map = {
                "巡检点1": "beijing",
                "巡检点2": "guangzhou",
                "巡检点3": "jilin",
                "巡检点4": "shanghai",
                "巡检点5": "shenzhen"
            }
            prefix = pavilion_map.get(name, "")

            if fire and not extinguisher:
                rospy.logwarn("[Task2] FIRE and NO EXTINGUISHER at %s!" % name)
                if prefix:
                    play_audio(f"{prefix}_fire_no_extinguisher.wav")
                anomalies.append(("火灾隐患", name))
                anomalies.append(("未放置灭火器", name))
            elif fire:
                rospy.logwarn("[Task2] FIRE detected at %s!" % name)
                if prefix:
                    play_audio(f"{prefix}_fire.wav")
                else:
                    play_audio("fire_detected.wav")
                anomalies.append(("火灾隐患", name))
            elif not extinguisher:
                rospy.logwarn("[Task2] NO extinguisher at %s!" % name)
                if prefix:
                    play_audio(f"{prefix}_no_extinguisher.wav")
                else:
                    play_audio("no_extinguisher.wav")
                anomalies.append(("未放置灭火器", name))

            if not fire and extinguisher:
                rospy.loginfo("[Task2] %s: all clear" % name)
                play_audio("all_clear.wav")

            time.sleep(0.5)

        return anomalies

    # ============================================================
    # AI视觉检测（接入 YOLO 检测结果）
    # ============================================================
    def detect_fire(self):
        """检测火灾隐患，返回 True/False"""
        return self._check_detection("fire")

    def detect_extinguisher(self):
        """检测灭火器是否放置到位，返回 True/False"""
        return self._check_detection("extinguisher")

    def _check_detection(self, keyword, timeout=3.0):
        """
        通过 YOLO 话题检查是否检测到指定物体
        在 timeout 秒内持续监听 /yolo/detections，看是否有匹配
        """
        # 先清空缓存
        self._last_detections = ""
        start = time.time()
        while time.time() - start < timeout and not rospy.is_shutdown():
            if hasattr(self, '_last_detections') and self._last_detections:
                if keyword in self._last_detections.lower():
                    rospy.loginfo("[Vision] Detected: %s" % keyword)
                    return True
            time.sleep(0.1)
        rospy.loginfo("[Vision] Did NOT detect: %s" % keyword)
        return False

    def _detection_cb(self, msg):
        """接收 YOLO 检测结果"""
        self._last_detections = msg.data

    # ============================================================
    # 自主对接充电（任务二阶段5 & 任务三打字/语音触发）
    # ============================================================
    def execute_ar_docking(self):
        """调用 AR 码识别和 relative_move 进行精准对接"""
        rospy.loginfo("[Docking] 等待服务 /track 启动...")
        try:
            rospy.wait_for_service('/track', timeout=5.0)
        except rospy.ROSException:
            rospy.logerr("[Docking] /track 服务未启动，对接失败")
            return False

        rospy.loginfo("[Docking] 等待服务 /relative_move 启动...")
        try:
            rospy.wait_for_service('/relative_move', timeout=5.0)
        except rospy.ROSException:
            rospy.logerr("[Docking] /relative_move 服务未启动，对接失败")
            return False

        track_client = rospy.ServiceProxy('/track', Track)
        relmove_client = rospy.ServiceProxy('/relative_move', SetRelativeMove)

        # 1. 寻找 AR 码并调整到 0.4 米
        req_track = TrackRequest()
        req_track.ar_id = 0
        req_track.goal_dist = 0.4
        rospy.loginfo("[Docking] 请求 /track 服务寻找 AR 码...")
        try:
            resp1 = track_client(req_track)
            if resp1.success:
                rospy.loginfo("[Docking] 二次定位成功：%s" % resp1.message)
            else:
                rospy.logerr("[Docking] 二次定位失败：%s" % resp1.message)
                return False
        except Exception as e:
            rospy.logerr("[Docking] /track 调用异常：%s" % e)
            return False

        # 2. 后退 0.18 米
        req_rel = SetRelativeMoveRequest()
        req_rel.goal.x = -0.18
        req_rel.goal.y = 0.0
        req_rel.goal.theta = 0.0
        req_rel.global_frame = "odom"
        rospy.loginfo("[Docking] 请求后退 0.18 米（对接充电）...")
        try:
            resp2 = relmove_client(req_rel)
            if resp2.success:
                rospy.loginfo("[Docking] 后退对接成功！")
            else:
                rospy.logerr("[Docking] 后退失败")
                return False
        except Exception as e:
            rospy.logerr("[Docking] /relative_move 调用异常：%s" % e)
            return False

        # 3. 停顿充电
        rospy.sleep(2.0)

        # 4. 前进 0.18 米（离开充电桩）
        req_rel.goal.x = 0.18
        rospy.loginfo("[Docking] 请求前进 0.18 米（脱离充电桩）...")
        try:
            resp3 = relmove_client(req_rel)
            if resp3.success:
                rospy.loginfo("[Docking] 前进脱离成功！")
            else:
                rospy.logerr("[Docking] 前进脱离失败")
                return False
        except Exception as e:
            rospy.logerr("[Docking] /relative_move 调用异常：%s" % e)
            return False

        return True

    def do_charging(self):
        """自动去充电桩并对接"""
        rospy.loginfo("[Task2/3] Heading to charging station...")
        play_audio("low_battery.wav")              # "电量过低，正在前往充电桩"

        # 1. 导航到充电桩大致位置
        cx, cy, cyaw = CHARGING_STATION
        success = self._navigate_with_reverse(cx, cy, cyaw)

        if success:
            rospy.loginfo("[Task2/3] Arrived at general charging area, starting AR docking...")
            # 2. 调用 AR 自动对接
            if self.execute_ar_docking():
                rospy.loginfo("[Task2/3] Charging dock reached and docked!")
                play_audio("charging_success.wav")     # TODO: 待生成 "充电对接成功"
            else:
                rospy.logwarn("[Task2/3] Docking failed!")
        else:
            rospy.logwarn("[Task2/3] Failed to reach general charging area")

    # ============================================================
    # 倒车入库导航工具集
    #   出馆转身 → 反向行驶 → 到达目标 yaw=0（摄像头朝外）
    #   避免入馆时转圈撞墙
    # ============================================================
    def get_current_yaw(self):
        """从 odom 四元数提取当前 yaw 角（弧度）"""
        if self.robot_orientation is None:
            return 0.0
        q = self.robot_orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _normalize_angle(self, angle):
        """将角度归一化到 [-π, π]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def rotate_in_place(self, target_yaw, angular_vel=0.6, tolerance=0.05, timeout=15):
        """
        原地旋转到目标 yaw（绝对角度，弧度）。
        使用 cmd_vel 直接控制，比例减速确保平滑到位。
        不经过 move_base，避免规划器在馆内规划出转圈路径。
        """
        rospy.loginfo("[Rotate] 开始旋转: 目标=%.2f, 当前=%.2f" % (target_yaw, self.get_current_yaw()))
        rate = rospy.Rate(20)

        start_time = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time < timeout):
            current = self.get_current_yaw()
            error = self._normalize_angle(target_yaw - current)

            if abs(error) < tolerance:
                # 到位，发送停止指令
                self.cmd_vel_pub.publish(Twist())
                rospy.loginfo("[Rotate] 旋转完成: 目标=%.2f, 实际=%.2f" % (target_yaw, current))
                return True

            twist = Twist()
            # 比例控制：远离目标全速，接近时减速（最低 0.15 rad/s 防卡死）
            speed = min(angular_vel, max(0.15, abs(error) * 0.8))
            twist.angular.z = speed if error > 0 else -speed
            self.cmd_vel_pub.publish(twist)
            rate.sleep()

        # 超时停止
        self.cmd_vel_pub.publish(Twist())
        rospy.logwarn("[Rotate] 旋转超时, 当前=%.2f" % self.get_current_yaw())
        return False

    def _navigate_with_reverse(self, x, y, yaw, timeout=NAV_TIMEOUT):
        """
        智能导航（倒车入库策略）：
        - 从起点出发：直接前进入馆（摄像头正对馆门）
        - 从馆内出发：先原地转 180° 让摄像头朝后，TEB 倒车行驶，
          沿途平滑将 yaw 过渡到目标值，到达时摄像头自然朝外
        """
        # if not self.at_origin:
        #     rospy.loginfo("[Nav] 出馆转身，准备倒车入库...")
        #     self.rotate_in_place(math.pi)

        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        if not client.wait_for_server(rospy.Duration(5)):
            rospy.logerr("[Nav] move_base 不可用")
            return False

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)

        rospy.loginfo("[Nav] 导航到 (%.2f, %.2f, yaw=%.2f)" % (x, y, yaw))
        
        # 增加死循环重试机制：找不到路就一直试，直到成功
        while not rospy.is_shutdown():
            client.send_goal(goal)
            finished = client.wait_for_result(rospy.Duration(timeout))

            if finished and client.get_state() == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("[Nav] 到达目标")
                return True
            else:
                rospy.logwarn("[Nav] 路径被阻挡或导航失败，等待 2 秒后重新规划...")
                client.cancel_goal()
                rospy.sleep(2.0)
                
        return False

    def navigate_to_venue(self, venue, timeout=NAV_TIMEOUT):
        """导航到指定场馆（自动处理倒车入库）"""
        if venue not in LOCATIONS:
            rospy.logwarn("[Nav] 未知场馆: %s" % venue)
            return False
        x, y, yaw = LOCATIONS[venue]
        rospy.loginfo("[Nav] 前往 %s" % venue)
        return self._navigate_with_reverse(x, y, yaw, timeout)

    def navigate_to_origin(self, timeout=NAV_TIMEOUT):
        """
        返回起点：先转身让摄像头朝后，TEB 沿途连贯地将 yaw 从 π 过渡到 0，
        到达起点时摄像头自然朝前（一步完成，不分两步）。
        """
        x, y, yaw = LOCATIONS["起点"]
        rospy.loginfo("[Nav] 返回起点")
        return self._navigate_with_reverse(x, y, yaw, timeout)

    # ============================================================
    # 任务三：训练模式
    #   打字控制机器人去指定位置停留
    # ============================================================
    def _cleanup_markers(self):
        """清除所有随机生成的标记"""
        from gazebo_msgs.srv import DeleteModel
        for i in range(1, 6):
            for prefix in ["fire_hazard", "fire_extinguisher"]:
                try:
                    rospy.wait_for_service('/gazebo/delete_model', timeout=3)
                    srv = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
                    srv(f"{prefix}_{i}")
                except:
                    pass

    def do_training_task(self):
        rospy.loginfo("[Task3] ========== 训练模式 ==========")
        print("""
  命令:
    goto <地点>  - 导航去指定位置（北京馆/广州馆/吉林馆/上海馆/深圳馆）
    spawn       - 随机生成火焰和灭火器标记
    clean       - 清除所有标记
    back        - 返回起点
    quit        - 退出训练模式，返回菜单
        """)
        rospy.loginfo("[Task3] ================================")

        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        import sys
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        def check_switch_cmd(text):
            """检查是否包含切换模式或充电桩命令"""
            if extract_switch_to_task2(text):
                rospy.loginfo("[Task3] Switch to task 2!")
                self.mode = 'admin'
                return True
            if "切换到任务一" in text or "切换任务一" in text or "任务一" in text or "参观模式" in text or "任务1" in text:
                rospy.loginfo("[Task3] Switch to task 1!")
                self.mode = 'visitor'
                return True
            if "充电桩" in text or "去充电" in text:
                rospy.loginfo("[Task3] Command '充电桩' received!")
                self.do_charging()
                return False # 执行完充电继续留在训练模式
            return False

        def handle_voice_switch():
            """非阻塞检查语音切换指令"""
            if self.voice_cmd:
                text = self.voice_cmd
                self.voice_cmd = None
                return check_switch_cmd(text)
            return False

        while not rospy.is_shutdown():
            # 检查语音切换
            if handle_voice_switch():
                break

            # 打字输入
            print("\n[Task3] > ", end="", flush=True)
            try:
                cmd = input().strip()
            except (EOFError, KeyboardInterrupt):
                break

            if handle_voice_switch():
                break

            # 解析命令：支持直接打地点名（北京馆）、goto北京馆、去北京馆
            if cmd in LOCATIONS:
                rospy.loginfo("[Task3] Going to: %s" % cmd)
                self.navigate_to_venue(cmd)
                rospy.loginfo("[Task3] Arrived, waiting here...")
            elif cmd.startswith("goto") or cmd.startswith("去"):
                place = cmd.replace("goto", "").replace("去", "").strip()
                if place in LOCATIONS:
                    rospy.loginfo("[Task3] Going to: %s" % place)
                    self.navigate_to_venue(place)
                    rospy.loginfo("[Task3] Arrived, waiting here...")
                else:
                    print("  未知地点: %s (可选: %s)" % (place, ", ".join(LOCATIONS.keys())))
            elif cmd == "quit" or cmd == "exit":
                break
            elif cmd == "spawn":
                import random_spawn
                random_spawn.main()
            elif cmd == "clean":
                self._cleanup_markers()
                rospy.loginfo("[Task3] All markers cleaned")
            elif cmd == "back":
                self.navigate_to_origin()
            elif check_switch_cmd(cmd):
                # 如果 check_switch_cmd 返回 True，说明切换了任务1/2，退出训练模式
                break
            elif cmd == "" or cmd == "help":
                print("  可用命令: goto, spawn, clean, back, quit")
            else:
                print("  未知命令: %s (输入 help 查看帮助)" % cmd)

        # 退出训练模式
        rospy.loginfo("[Task3] Exiting training mode...")
        self._cleanup_markers()
        self.navigate_to_origin()
        rospy.loginfo("[Task3] Training mode ended")
if __name__ == '__main__':
    rospy.init_node('competition_task')
    rospy.loginfo("PKG_DIR: " + PKG_DIR)

    print("\n" + "="*40)
    print("       比赛任务选择")
    print("="*40)
    print(" 1. 参观模式 - 人脸唤醒 + 语音导航")
    print(" 2. 管理模式 - 人脸识别管理员")
    print(" 3. 训练模式 - 纯打字控制")
    print("="*40)

    choice = ""
    while choice not in ('1', '2', '3'):
        try:
            choice = input("请选择模式 (1/2/3): ").strip()
        except:
            choice = "1"

    mode_map = {'1': 'visitor', '2': 'admin', '3': 'train'}
    mode = mode_map[choice]
    node = CompetitionTask(mode=mode)
    node.run()
