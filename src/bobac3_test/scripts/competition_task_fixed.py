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

# 入库方向说明：yaw=0 时摄像头朝外（朝向走道）
# TEB 已开启 allow_init_with_backwards_motion，规划器自动倒车入库停在此朝向
LOCATIONS = {
    "起点":  (0.0,  -0.2, 0.0),
    "北京馆": (2.55, 0.1,  0.0),
    "广州馆": (2.55, 1.1,  0.0),
    "吉林馆": (2.55, 2.1,  0.0),
    "深圳馆": (1.05, 1.1,  0.0),
    "上海馆": (1.05, 2.1,  0.0),
}

# 出库点：库门口外的走道中央
# 对应 x=2.55 列场馆出口在左，x=1.05 列在左
VENUE_EXIT_POINTS = {
    "北京馆": (1.9, 0.1),
    "广州馆": (1.9, 1.1),
    "吉林馆": (1.9, 2.1),
    "深圳馆": (0.5, 1.1),
    "上海馆": (0.5, 2.1),
}

# 任务二岋細巡检点癸紙5涓鎸囧畾浣嶇疆锛岄『搴忓贰妫锛屽叏閮 yaw=0锛
INSPECTION_POINTS = [
    ("巡检点1", 2.55, 0.1,  0.0),
    ("巡检点2", 2.55, 1.1,  0.0),
    ("巡检点3", 2.55, 2.1,  0.0),
    ("巡检点4", 1.05, 2.1,  0.0),
    ("巡检点5", 1.05, 1.1,  0.0),
]

# 充电桩位置（起点处）
CHARGING_STATION = (0.0, -0.2, 0.0)

# 巡检指令关键字
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
    return name.rstrip("棣")

def play_audio(filename):
    """播报语音"""
    path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(path):
        subprocess.run(['aplay', '-q', path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    rospy.logwarn("Audio not found: " + filename)
    return False

def extract_place(text):
    """浠庤闊虫寚浠や腑鎻愬彇鐩鏍囧満棣"""
    if not text:
        return None

    # 瀹屾暣鍖归厤
    for venue in ["广州馆", "吉林馆", "北京馆", "深圳馆", "上海馆"]:
        if venue in text:
            return venue

    # 简称匹配
    short = {"骞垮窞":"广州馆", "鍚夋灄":"吉林馆", "鍖椾含":"北京馆", "娣卞湷":"深圳馆", "涓婃捣":"上海馆"}
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

        # 仅在初始化麦克风时屏蔽 ALSA 噪音，初始化完就恢复
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

        rospy.loginfo("[Mic] 识别结果: %s" % (result if result else "(绌)"))
        return result

class CompetitionTask:
    def __init__(self, mode='visitor'):
        """
        mode: 'visitor' # 任务一：参观模式，人脸唤醒后语音导航
              'admin'   鈥 任务二岋細管理模式锛岃瘑鍒浜鸿劯鍒ゆ柇绠＄悊鍛
              'train'   # 任务三：训练模式，打字控制 + 语音切换，跳过人脸
        """
        self.mode = mode
        self.face_flag = False
        self.recognized_name = ""   # 识别到的中文名
        self.is_admin = False       # 是否为管理员
        self.current_venue = None
        self.voice_cmd = None       # Windows 端发来的语音指令
        self.robot_x = 0.0          # 机器人当前位置
        self.robot_y = 0.0
        self.at_origin = True       # 是否在原点

        rospy.Subscriber('/face_detection', face_results, self.face_cb)
        rospy.Subscriber('/voice_command', String, self.voice_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/yolo/detections', String, self._detection_cb)
        self._last_detections = ""

        # 通知 Windows 端开始/停止浜鸿劯妫娴
        self.face_trigger = rospy.Publisher('/face_trigger', String, queue_size=1)
        # 控制行驶朝向锁定鍦 yaw=0
        self.yaw_lock_pub = rospy.Publisher('/yaw_lock_active', Bool, queue_size=1)
        # 强制刹车底盘速度发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        time.sleep(0.5)  # 等 publisher 注册完成

        # 默认锁定 yaw 涓 0.0
        self.yaw_lock_pub.publish(Bool(data=True))

        rospy.loginfo("Mode: %s, ready" % mode)

    def odom_cb(self, msg):
        """璺熻釜机器人当前位置锛屽垽鏂是否在原点"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        dist = (self.robot_x**2 + self.robot_y**2)**0.5
        self.at_origin = dist < 0.5

    def face_cb(self, msg):
        """鎺ユ敹 Windows 绔鍙戞潵鐨勪汉鑴告娴嬬粨鏋"""
        if self.face_flag:
            return  # 宸茬粡鏀跺埌缁撴灉锛屽拷鐣ラ噸澶
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
        """鎺ユ敹 Windows 绔鍙戞潵鐨勮闊宠瘑鍒缁撴灉"""
        self.voice_cmd = msg.data
        rospy.loginfo("[Voice] 鏀跺埌: %s" % msg.data)

    def wait_for_voice(self, prompt=""):
        """绛夊緟 Windows 绔 /voice_command 璇闊虫寚浠わ紙闃诲炵洿鍒版敹鍒帮級"""
        rospy.loginfo("\n" + "="*30 + " >>> 璇疯磋瘽 (%s) <<< " % prompt + "="*30)

        self.voice_cmd = None

        while self.voice_cmd is None and not rospy.is_shutdown():
            time.sleep(0.1)

        if rospy.is_shutdown():
            return ""

        result = self.voice_cmd
        self.voice_cmd = None
        rospy.loginfo("[Voice] 缁撴灉: %s" % result)
        return result

    def run(self):
        rate = rospy.Rate(10)

        # 任务三夛紙训练模式忥級璺宠繃浜鸿劯妫娴嬶紝鐩存帴杩
        if self.mode == 'train':
            self.do_training_task()
            return

        while not rospy.is_shutdown():
            # ===== 1. 绛夊緟鍥炲埌原点 =====
            rospy.loginfo("[Run] 等待回到原点...")
            while not self.at_origin and not rospy.is_shutdown():
                rate.sleep()
            if rospy.is_shutdown():
                break

            # ===== 2. 瑙﹀彂 Windows 绔浜鸿劯妫娴嬶紙浼樺寲闃叉姈鏈哄埗锛=====
            rospy.loginfo("[Run] 已在原点，触发人脸检测 (%s 妯″紡)" % self.mode)
            self.face_flag = False
            self.recognized_name = ""
            self.is_admin = False

            # 銆愭牳蹇冧慨鏀圭偣銆戯細只发送一次触发信号，不再使用短周期的 while 杞扮偢
            self.face_trigger.publish(String(data=self.mode))
            rospy.loginfo("[Run] 宸插崟娆″彂閫 /face_trigger: %s锛岀瓑寰 Windows 绔鍝嶅簲..." % self.mode)
            
            # 设置最大等待时间为 15 秒掞紝鏈熼棿姣 0.1 绉掓鏌ヤ竴娆 Windows 鐨勮繑鍥炵姸鎬
            wait_start = time.time()
            while not self.face_flag and (time.time() - wait_start < 15.0) and not rospy.is_shutdown():
                time.sleep(0.1)
                
            if rospy.is_shutdown():
                break

            # 如果在 15 秒内没有浠讳綍浜鸿劯鍙嶉堬紝寮鸿屽彂閫 stop 鎸囦护閲嶇疆 Windows 骞堕噸鏂拌繘鍏ュ師鐐圭瓑寰
            if not self.face_flag:
                rospy.logwarn("[Run] 触发人脸检测超时讹紙鏈瑙佷汉鑴革級锛屽彂閫 stop 淇″彿娓呯┖ Windows 鐘舵")
                self.face_trigger.publish(String(data='stop'))
                time.sleep(1.0)
                continue

            # 成功接收到颁汉鑴镐俊鍙峰悗锛屽皢鏍囧織浣嶅綊闆
            self.face_flag = False
            
            # 銆愰噸瑕佸畨鍏ㄥ垏鏂銆戯細立刻向 Windows 端补发一个 'stop' 信号，清除 ROS 队列积压并安抚 Windows 端
            self.face_trigger.publish(String(data='stop'))
            rospy.loginfo("[Run] 成功获取人脸数据，已发送 'stop' 信号关闭 Windows 端人脸检测")

            # ===== 3. 任务三锛氳垮㈡ā寮忥紙浠绘剰浜鸿劯锛=====
            if self.mode == 'visitor':
                self.do_visitor_task()

                # 检查是否在任务涓涓鍒囨崲鍒颁簡任务二
                if self.mode == 'admin':
                    rospy.loginfo("[Run] Mode switched to admin, re-triggering face detection...")
                    continue
                elif self.mode == 'train':
                    rospy.loginfo("[Run] Mode switched to training, entering task 3...")
                    self.do_training_task()
                    # 训练模式结束熷悗鍥炲埌寰鐜
                    self.mode = 'visitor'
                    continue    # 鍥炲埌寰鐜椤堕儴锛岄噸鏂拌Е鍙戜汉鑴告娴嬶紙绠＄悊鍛樿瘑鍒锛

            # ===== 4. 任务二岋細管理模式锛堥渶瑕佺＄悊鍛橈級=====
            elif self.mode == 'admin':
                if self.is_admin and self.recognized_name:
                    self.do_admin_task()
                else:
                    rospy.logwarn("[Run] 非注册管理员，拒绝执行任务，1秒后重新进入等待...")
                    time.sleep(1.0)

                # 检查是否切换㈠埌浜嗚缁冩ā寮
                if self.mode == 'train':
                    rospy.loginfo("[Run] Admin switched to training, entering task 3...")
                    self.do_training_task()
                    self.mode = 'visitor'
                    continue

            rate.sleep()

    # ============================================================
    # 任务三锛氳繋瀹惧艰堟ā寮
    #   璇勫垎锛氫汉鑴稿敜閱30 + 璇闊充氦浜10 + 瀵艰堝艰埅40 + 杩斿洖鍑哄彂鍖30 = 110鍒
    # ============================================================
    def do_visitor_task(self):
        # ===== 阶段1：人脸检测唤醒（30分）=====
        # 鎾鎶ワ細"浣犲ソ锛屾㈣繋鎮ㄧ殑鍒版潵锛佹湁浠涔堥渶瑕佸府鍔╃殑鍚楋紵"
        rospy.loginfo("[Task1] Face detected, activating...")
        play_audio("welcome.wav")
        time.sleep(1.0)

        # ===== 阶段2：语音交互应答（10分）=====
        # 正确识别"鍙浠ュ甫鎴戝幓鍙傝俋X棣嗗悧锛"绛夎嚜鐒惰闊虫寚浠
        venue = None
        for attempt in range(3):
            text = self.wait_for_voice(prompt="attempt %d/3" % (attempt+1))

            rospy.loginfo("[Task1] Heard: %s" % text)

            # 妫娴嬫槸鍚﹁佸垏鎹㈠埌任务二
            if extract_switch_to_task2(text):
                rospy.loginfo("[Task1] Switch to task 2 requested!")
                play_audio("switch_task2.wav")
                self.mode = 'admin'
                return

            # 妫娴嬫槸鍚﹁佸垏鎹㈠埌任务三
            if extract_switch_to_task3(text):
                rospy.loginfo("[Task1] Switch to task 3 (training) requested!")
                self.mode = 'train'
                return

            venue = extract_place(text)

            if venue:
                rospy.loginfo("[Task1] Venue recognized: " + venue)
                self.current_venue = venue
                # 纭璁ゆ挱鎶ワ細"濂界殑锛屽甫鎮ㄥ幓XX棣"
                play_audio("confirm_" + strip_guan(venue) + ".wav")
                break
            else:
                rospy.logwarn("[Task1] No venue in: %s" % text)
                if attempt < 2:
                    # 娌″惉娓咃紝鎻愮ず鍐嶈翠竴娆
                    play_audio("retry_command.wav")
                    time.sleep(0.3)

        if not venue:
            rospy.logwarn("[Task1] Failed to recognize venue after 3 attempts")
            return

        # ===== 阶段3：迎宾引导导航（40分）=====
        nav_success = self.go_to_venue(venue)
        if not nav_success:
            rospy.logwarn("[Task1] Navigation to %s failed" % venue)
            # 瀵艰埅澶辫触涔熷皾璇曡繑鍥
            send_nav_goal("起点")
            return

        # 到达后播放场馆介绍
        rospy.loginfo("[Task1] Reached %s, playing introduction..." % venue)
        play_audio("intro_" + strip_guan(venue) + ".wav")
        time.sleep(1.5)

        # ===== 阶段4：返回出发区（30分）=====
        # 播报归位语音并导航返回起点（导航10 + 未压线10 + 语音10）
        back_file = "back_" + venue + ".wav"
        if not os.path.exists(os.path.join(AUDIO_DIR, back_file)):
            back_file = "back.wav"                 # fallback 閫氱敤杩旂▼璇闊
        rospy.loginfo("[Task1] Returning: " + back_file)
        play_audio(back_file)
        time.sleep(0.5)

        rospy.loginfo("[Task1] Navigating back to start...")
        send_nav_goal("起点")

        # 閲嶇疆鐘舵
        self.recognized_name = ""
        self.is_admin = False
        self.current_venue = None
        rospy.loginfo("[Task1] Mission complete!")

    # ============================================================
    # 任务二岋細管理模式
    #   人脸识别管理员 鈫 闅忔満鐢熸垚鏍囪 鈫 璇闊虫寚浠 鈫 澶氱偣巡检 鈫 寮傚父妫娴 鈫 充电
    # ============================================================
    def do_admin_task(self):
        # -------- 闃舵1锛氱＄悊鍛樿瘑鍒 --------
        if self.is_admin and self.recognized_name:
            rospy.loginfo("[Task2] Admin recognized: %s" % self.recognized_name)

            # 鎾鎶ワ細"娆㈣繋绠＄悊鍛榅XX杩涘叆绯荤粺"
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

        # -------- 闃舵1.5锛氶殢鏈虹敓鎴愮伃鐏鍣ㄥ拰火焰版爣璁 --------
        rospy.loginfo("[Task2] Spawning random fire/extinguisher markers...")
        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        import sys
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import random_spawn
        random_spawn.main()          # 闅忔満鏀剧疆锛岃嚜鍔ㄦ竻闄や笂涓杞

        # -------- 闃舵2锛氱瓑寰呭贰妫鎸囦护 --------
        rospy.loginfo("[Task2] Waiting for inspection command...")
        play_audio("wait_command.wav")             # "璇蜂笅杈惧贰妫鎸囦护"

        cmd_received = False
        for attempt in range(3):
            text = self.wait_for_voice(prompt="巡检指令 %d/3" % (attempt+1))

            # 妫娴嬫槸鍚﹁佸垏鎹㈠埌任务三
            if extract_switch_to_task3(text):
                rospy.loginfo("[Task2] Switch to task 3 (training) requested!")
                self.mode = 'train'
                return

            if extract_command(text):
                cmd_received = True
                rospy.loginfo("[Task2] Inspection command received: " + text)
                play_audio("inspection_start.wav") # "鏀跺埌鎸囦护锛屽紑濮嬫墽琛屽贰妫浠诲姟"
                break
            else:
                rospy.loginfo("[Task2] Retry %d: %s" % (attempt+1, text))
                if attempt < 2:
                    play_audio("retry_command.wav")  # "璇峰啀璇翠竴娆"

        if not cmd_received:
            rospy.logwarn("[Task2] No inspection command received")
            play_audio("inspection_cancel.wav")    # "鏈鏀跺埌鎸囦护锛屽彇娑堝贰妫"
            self.recognized_name = ""
            self.is_admin = False
            return

        # -------- 闃舵3锛氬氱偣鑷涓诲贰妫 --------
        anomalies = self.do_inspection()

        # -------- 闃舵4锛氬贰妫瀹屾垚锛屾挱鎶ョ粨鏋 --------
        rospy.loginfo("[Task2] Inspection complete, anomalies: %d" % len(anomalies))
        if len(anomalies) > 0:
            play_audio("inspection_anomaly_summary.wav")  # TODO: 寰呯敓鎴
            for a in anomalies:
                rospy.logwarn("  Anomaly: %s at %s" % (a[0], a[1]))
        else:
            play_audio("inspection_complete.wav")  # "巡检浠诲姟瀹屾垚锛屼竴鍒囨ｅ父"

        # -------- 闃舵5锛氳嚜涓诲规帴充电 --------
        self.do_charging()

        # 閲嶇疆鐘舵
        self.recognized_name = ""
        self.is_admin = False
        rospy.loginfo("[Task2] Mission complete!")

    # ============================================================
    # 澶氱偣巡检锛堜换鍔′簩闃舵3锛
    #   渚濇″艰埅鍒5涓巡检点癸紝姣忎釜鐐规娴嬪紓甯
    # ============================================================
    def do_inspection(self):
        anomalies = []

        # 鍒涘缓涓涓瀵艰埅瀹㈡埛绔锛屾墍鏈夊贰妫鐐瑰嶇敤
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        if not client.wait_for_server(rospy.Duration(5)):
            rospy.logerr("[Task2] move_base not available")
            return anomalies

        for i, (name, x, y, yaw) in enumerate(INSPECTION_POINTS):
            rospy.loginfo("[Task2] Inspecting: %s (%.1f, %.1f, yaw=%.2f)" % (name, x, y, yaw))

            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = "map"
            goal.target_pose.header.stamp = rospy.Time.now()
            goal.target_pose.pose.position.x = x
            goal.target_pose.pose.position.y = y
            goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)

            client.send_goal(goal)
            finished = client.wait_for_result(rospy.Duration(NAV_TIMEOUT))
            state = client.get_state()

            rospy.loginfo("[Task2] %s result: finished=%s, state=%d (SUCCEEDED=%d)" % (name, finished, state, actionlib.GoalStatus.SUCCEEDED))

            if not finished or state != actionlib.GoalStatus.SUCCEEDED:
                rospy.logwarn("[Task2] Failed to reach %s (state=%d), skipping" % (name, state))
                continue

            rospy.loginfo("[Task2] Reached %s" % name)
            play_audio("point_%s.wav" % name)

            # AI瑙嗚夊紓甯告娴
            fire = self.detect_fire()
            extinguisher = self.detect_extinguisher()

            if fire:
                rospy.logwarn("[Task2] FIRE detected at %s!" % name)
                play_audio("fire_detected.wav")
                anomalies.append(("火灾隐患", name))

            if not extinguisher:
                rospy.logwarn("[Task2] NO extinguisher at %s!" % name)
    # ===========================================================
    # AI视觉市常짃测 (接入 YOLO 检测结果)
    # ===========================================================
    def detect_fire(self):
        return self._check_detection("fire")

    def detect_extinguisher(self):
        return self._check_detection("fire extinguisher")

    def _check_detection(self, keyword, timeout=3.0):
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
        self._last_detections = msg.date

    def _send_goal_raw(self, client, x, y, yaw, timeout=30):
        """(x, y, yaw) 鐩鏍囧艰埅銆傜粨鍚 move_base 瑙勫垝涓庡簳灞傚姩鎬侀攣澶达紝瀹炵幇鍑哄簱杈瑰紑杈硅嚜杞锛屽埌浜嗚蛋寤婄粷瀵归攣瀹 yaw=0 鍊掗骞崇Щ鍏ュ簱"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation = yaw_to_quaternion(yaw)
        
        import tf
        from tf.transformations import euler_from_quaternion
        tf_listener = tf.TransformListener()
        time.sleep(0.2) # 绋嶇瓑 TF 缂撳瓨

        # 瀹氫箟璧板粖鐨勫垎鐣岀嚎
        # 濡傛灉鍘 鍖椾含/骞垮窞/鍚夋灄 (x=2.55)锛岃蛋寤婂湪 x=1.9锛涘傛灉鍘 娣卞湷/涓婃捣 (x=1.05)锛岃蛋寤婂湪 x=0.5
        x_corridor = 1.95 if x > 1.8 else 0.55
        
        # 璁板綍閿佸畾鐘舵
        self.lock_active = False

        # 杈呭姪鍑芥暟锛氳幏鍙栧疄鏃 yaw
        def get_current_yaw():
            try:
                (trans, rot) = tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
                _, _, cy = euler_from_quaternion(rot)
                return trans[0], cy
            except:
                return self.robot_x, 0.0

        # 搴曞眰閫熷害鎷︽埅涓庢帶鍒堕昏緫
        def raw_cmd_cb(msg):
            filter_msg = Twist()
            filter_msg.linear.x = msg.linear.x
            filter_msg.linear.y = msg.linear.y
            filter_msg.linear.z = msg.linear.z
            filter_msg.angular.x = msg.angular.x
            filter_msg.angular.y = msg.angular.y
            
            cx, cyaw = get_current_yaw()
            
            # 鍒ゆ柇鏄鍚﹀簲璇ュ紑鍚濮挎侀攣锛
            # 濡傛灉鏈哄櫒浜哄凡缁忓湪璧板粖涓 (x 鍧愭爣宸茬粡灏忎簬璧板粖鐨勯槇鍊肩嚎)锛屽垯寮哄埗杩涘叆閿佸畾鐘舵
            if cx <= x_corridor:
                if not self.lock_active:
                    rospy.loginfo("[Nav] 鏈哄櫒浜哄凡鍒拌揪璧板粖 (x=%.2f)锛屽紑鍚 yaw=0 寮哄姏闂鐜閿佸ご..." % cx)
                    self.lock_active = True
            
            if self.lock_active:
                # 閿佸ご婵娲伙細灏嗚掗熷害鏇挎崲涓 PD 闭环绾犲亸瑙掗熷害锛岄攣姝绘湞鍚戜负 0.0 搴
                yaw_err = 0.0 - cyaw
                while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
                while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
                
                # kp = 2.5
                filter_msg.angular.z = 2.5 * yaw_err
            else:
                # 閿佸ご鏈婵娲伙紙杩樺湪棣嗗唴鎴栧垰鍑洪嗭級锛氬師鏍疯浆鍙戣掗熷害锛屽厑璁 move_base 鍋氳嚜杞/鐢╁熬鍑哄簱鍔ㄤ綔
                filter_msg.angular.z = msg.angular.z
                
            self.cmd_vel_pub.publish(filter_msg)
            
        raw_cmd_sub = rospy.Subscriber('/cmd_vel_raw', Twist, raw_cmd_cb)
        
        rospy.loginfo("[Nav] 鍙戦佺洰鏍囩偣: (%.2f, %.2f)锛屽姩鎬佸嚭搴撹嚜杞涓庤蛋寤婇攣澶村凡婵娲...", x, y)
        client.send_goal(goal)
        
        start_time = time.time()
        reached = False
        aborted = False

        rate = rospy.Rate(20) # 20Hz 鎺ヨ繎涓庣姸鎬佺洃鎺
        while (time.time() - start_time < timeout) and not rospy.is_shutdown():
            state = client.get_state()
            
            # 璁＄畻褰撳墠鐗╃悊璺濈
            dx = self.robot_x - x
            dy = self.robot_y - y
            dist = math.sqrt(dx*dx + dy*dy)
            
            # 璺濈 12 鍘樼背浠ュ唴锛岀洿鎺ヤ汉宸ユ埅鏂瀵艰埅骞跺埞杞︼紝缁濅笉鎸ｆ墡
            if dist < 0.12:
                rospy.loginfo("[Nav] 璺濈荤洰鏍囦粎鏈 %.3f 绫筹紝宸插埌杈撅紒鎵撴柇瀵艰埅銆" % dist)
                client.cancel_goal()
                reached = True
                break
                
            if state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("[Nav] move_base 鑷琛屾姤鍛婂埌杈")
                reached = True
                break
            elif state in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
                rospy.logwarn("[Nav] move_base 瑙勫垝澶辫触鎴栫粓姝 (state=%d)" % state)
                aborted = True
                break
                
            rate.sleep()

        # 娉ㄩ攢閫熷害鍥炶皟鐩戝惉鍣
        raw_cmd_sub.unregister()

        # 鏈缁堝埞杞︿笌寰璋冿紝纭淇濆仠寰楃ǔ涓旇掑害鏋佸害绮剧‘
        stop_msg = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.06)
            
        # 鏈鍚庣殑濮挎佸井璋冨归綈
        adjust_start = time.time()
        while time.time() - adjust_start < 0.3 and not rospy.is_shutdown():
            _, cyaw = get_current_yaw()
            yaw_err = 0.0 - cyaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            
            if abs(yaw_err) > 0.026:
                msg = Twist()
                msg.angular.z = max(min(2.0 * yaw_err, 0.15), -0.15)
                self.cmd_vel_pub.publish(msg)
            else:
                break
            rate.sleep()

        # 閿佹
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_msg)
            time.sleep(0.05)

        return reached and not aborted
        if x > 1.8: # 閽堝 鍖椾含/骞垮窞/吉林馆 (x=2.55)锛岃蛋寤婂湪 x=1.9
            x_exit = 1.90
        else: # 閽堝 娣卞湷/上海馆 (x=1.05)锛岃蛋寤婂湪 x=0.5
            x_exit = 0.50

        # 濡傛灉鎴戜滑鐜板湪鏄鍦ㄦ煇涓鍦洪嗛噷 (姣斿 x > 1.95 鎴 x > 0.6)
        if (x > 1.8 and cur_x > 1.95) or (x < 1.8 and cur_x > 0.6):
            rospy.loginfo("[Nav] 阶段 1/3: 平移出库中...")
            while time.time() - start_time < timeout and not rospy.is_shutdown():
                cx, cy, cyaw = get_pose()
                if cx <= x_exit:
                    break # 宸插嚭鍒拌蛋閬
                
                # 鍙戝竷妯鍚戝線宸﹀钩绉婚熷害 (鍑忓皬x)锛屼笖闂鐜閿佹诲亸鑸瑙
                msg = Twist()
                msg.linear.x = -lateral_speed
                
                # 濮挎佺籂鍋
                yaw_err = 0.0 - cyaw
                while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
                while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
                msg.angular.z = kp_yaw * yaw_err
                
                cmd_pub.publish(msg)
                rate.sleep()

        # ========== 闃舵 2: 璧板粖鐩磋屽钩绉诲埌鐩鏍囬嗛棬鍙 ==========
        rospy.loginfo("[Nav] 阶段 2/3: 走道直行平移中...")
        while time.time() - start_time < timeout and not rospy.is_shutdown():
            cx, cy, cyaw = get_pose()
            dy = y - cy
            
            # y 鍧愭爣对齐鍒 0.05 绫充互鍐呭嵆璁や负鍒拌揪棣嗛棬鍙
            if abs(dy) < 0.05:
                break
                
            msg = Twist()
            # 娌跨潃璧板粖骞崇Щ鐩磋 (鍦ㄥ湴鍥句笂鏄 y 鏂瑰悜)
            msg.linear.y = math.copysign(linear_speed, dy)
            
            # 濮挎佺籂鍋
            yaw_err = 0.0 - cyaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            msg.angular.z = kp_yaw * yaw_err
            
            cmd_pub.publish(msg)
            rate.sleep()

        # ========== 闃舵 3: 妯绉诲掕溅鍏ュ簱鍒版渶缁堢洰鏍囩偣 (x, y) ==========
        rospy.loginfo("[Nav] 阶段 3/3: 横移倒车入库中...")
        while time.time() - start_time < timeout and not rospy.is_shutdown():
            cx, cy, cyaw = get_pose()
            dx = x - cx
            
            # x 鍧愭爣对齐鍒 0.08 绫充互鍐呭嵆璁や负鍏ュ簱瀹屾垚
            if abs(dx) < 0.08:
                break
                
            msg = Twist()
            # 妯鍚戝線鍙冲钩绉 (澧炲姞 x 鍧愭爣)锛屽掕溅鍏ュ簱
            msg.linear.x = math.copysign(lateral_speed, dx)
            
            # 濮挎佺籂鍋
            yaw_err = 0.0 - cyaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            msg.angular.z = kp_yaw * yaw_err
            
            cmd_pub.publish(msg)
            rate.sleep()

        # ========== 阶段 4: 静态姿态锁死 ==========
        rospy.loginfo("[Nav] 到达目的地，强力刹车并微调姿态...")
        stop_msg = Twist()
        for _ in range(8):
            cmd_pub.publish(stop_msg)
            time.sleep(0.05)

        # 闭环微调 0.3 秒，确保角度绝对为 0
        adjust_start = time.time()
        while time.time() - adjust_start < 0.3 and not rospy.is_shutdown():
            _, _, cyaw = get_pose()
            yaw_err = 0.0 - cyaw
            while yaw_err > math.pi: yaw_err -= 2.0 * math.pi
            while yaw_err < -math.pi: yaw_err += 2.0 * math.pi
            
            if abs(yaw_err) > 0.02:
                msg = Twist()
                msg.angular.z = max(min(kp_yaw * yaw_err, 0.15), -0.15)
                cmd_pub.publish(msg)
            else:
                break
            rate.sleep()

        # 鏈缁堝埞杞
        for _ in range(5):
            cmd_pub.publish(stop_msg)
            time.sleep(0.05)

        rospy.loginfo("[Nav] 纯平移导航任务完成！")
        return True

    def go_to_venue(self, venue, timeout=NAV_TIMEOUT):
        """
        全向车一气痕成入库流程：
        鐩存帴灏嗙洰鏍囧満棣嗭紙yaw=0锛夊彂閫佺粰 move_base銆
        在 cmd_vel_filter 锁定 yaw=0 的配合下，机器人会直接自主平移出库、走道平移、并平移倒车入库，屽疄鐜颁竴鏉￠緳绉诲姩銆
        """
        if venue not in LOCATIONS:
            return False

        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        if not client.wait_for_server(rospy.Duration(5)):
            rospy.logerr("[Nav] move_base 不可用")
            return False

        x, y, yaw = LOCATIONS[venue]
        rospy.loginfo("[Nav] 一条龙导航前往 %s -> (%.2f, %.2f, yaw=0)" % (venue, x, y))
        success = self._send_goal_raw(client, x, y, yaw, timeout=timeout)

        if success:
            self.current_venue = venue
            rospy.loginfo("[Nav] 已成功到达并入库: %s" % venue)
        return success

    # ============================================================
    # 任务三夛細训练模式
    #   打字控制机器人去指定位置停留，支持语音"切换到任务三"
    # ============================================================
    def _cleanup_markers(self):
        """清除所有夐殢鏈虹敓鎴愮殑鏍囪"""
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
    spawn       - 随机生成火焰 and 灭火器标记
    clean       - 清除所有标记
    back        - 返回起点
    quit        - 退出训练模式，返回菜单
  语音:
    说 "切换到任务鍔′竴" 鎴 "切换到任务二" 鍒囨崲妯″紡
        """)
        rospy.loginfo("[Task3] ================================")

        scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../scripts")
        scripts_dir = os.path.abspath(scripts_dir)
        import sys
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        def handle_voice_switch():
            """非阻塞检查ヨ闊冲垏鎹㈡寚浠"""
            if self.voice_cmd:
                text = self.voice_cmd
                self.voice_cmd = None
                if extract_switch_to_task2(text):
                    rospy.loginfo("[Task3] Voice switch to task 2!")
                    self.mode = 'admin'
                    return True
                if "鍒囨崲鍒颁换鍔′竴" in text or "切换任务三" in text or "任务三" in text or "参观模式" in text:
                    rospy.loginfo("[Task3] Voice switch to task 1!")
                    self.mode = 'visitor'
                    return True
            return False

        while not rospy.is_shutdown():
            # 妫鏌ヨ闊冲垏鎹
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

            # 解析命令锛氭敮鎸佺洿鎺ユ墦地点鍚嶏紙北京馆嗭級銆乬oto北京馆嗐佸幓北京馆
            if cmd in LOCATIONS and cmd != "起点":
                rospy.loginfo("[Task3] Going to: %s" % cmd)
                self.go_to_venue(cmd)
                rospy.loginfo("[Task3] Arrived, waiting here...")
            elif cmd.startswith("goto") or cmd.startswith("鍘"):
                place = cmd.replace("goto", "").replace("鍘", "").strip()
                if place in LOCATIONS and place != "起点":
                    rospy.loginfo("[Task3] Going to: %s" % place)
                    self.go_to_venue(place)
                    rospy.loginfo("[Task3] Arrived, waiting here...")
                else:
                    print("  鏈鐭ュ湴鐐: %s (可选: %s)" % (place, ", ".join(LOCATIONS.keys())))
            elif cmd == "quit" or cmd == "exit":
                break
            elif cmd == "spawn":
                import random_spawn
                random_spawn.main()
            elif cmd == "clean":
                self._cleanup_markers()
                rospy.loginfo("[Task3] All markers cleaned")
            elif cmd == "back":
                send_nav_goal("起点")
            elif cmd == "" or cmd == "help":
                print("  可用命令: goto, spawn, clean, back, quit")
                print("  语音切换: 鍒囨崲鍒颁换鍔′竴 / 切换到任务二 / 切换到任务三")
            else:
                print("  鏈鐭ュ懡浠: %s (杈撳叆 help 鏌ョ湅帮助)" % cmd)

        # 退出训练模式
        rospy.loginfo("[Task3] Exiting training mode...")
        self._cleanup_markers()
        send_nav_goal("起点")
        rospy.loginfo("[Task3] Training mode ended")
if __name__ == '__main__':
    rospy.init_node('competition_task')
    rospy.loginfo("PKG_DIR: " + PKG_DIR)

    print("\n" + "="*40)
    print("             比赛任务选择")
    print("="*40)
    print(" 1. 1. 参观模式 - 人脸唤醒 + 语音导航")
    print(" 2. 2. 管理模式 - 人脸识别管理员")
    print(" 3. 3. 训练模式 - 打字控制 + 语音切换")
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
