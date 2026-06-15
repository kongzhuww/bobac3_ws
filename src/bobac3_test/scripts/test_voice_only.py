#!/usr/bin/env python3
import rospy
import actionlib
import subprocess
import json
import os
import time
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

LOCATIONS = {
    "起点": (0.0, 0.0), "广州馆": (2.6, 1.1), "吉林馆": (2.6, 2.2),
    "北京馆": (2.6, 0.1), "深圳馆": (1.2, 1.1), "上海馆": (1.2, 2.2),
}

PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AUDIO_DIR = os.path.join(PKG_DIR, "audio")
MODEL_DIR = os.path.join(PKG_DIR, "models", "vosk-model-small-cn-0.22")
SAMPLE_RATE = 16000
LISTEN_TIMEOUT = 15
NAV_TIMEOUT = 90

def play_audio(filename):
    path = os.path.join(AUDIO_DIR, filename)
    if os.path.exists(path):
        subprocess.run(['aplay', '-q', path])
        return True
    return False

def extract_place(text):
    if not text:
        return None
    for venue in ["广州馆", "吉林馆", "北京馆", "深圳馆", "上海馆", "起点"]:
        if venue in text:
            return venue
    short = {"广州":"广州馆", "吉林":"吉林馆", "北京":"北京馆", "深""":"深圳馆", "上海":"上海馆"}
    for k, v in short.items():
        if k in text:
            return v
    return None

def send_nav_goal(place, timeout=NAV_TIMEOUT):
    if place not in LOCATIONS:
        return False
    x, y = LOCATIONS[place]
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    if not client.wait_for_server(rospy.Duration(5)):
        rospy.logwarn("move_base not available")
        return False
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0
    client.send_goal(goal)
    return client.wait_for_result(rospy.Duration(timeout))

class SpeechRecognizer:
    def __init__(self, model_dir):
        import vosk
        vosk.SetLogLevel(-1)
        self.model = vosk.Model(model_dir)
    def listen_once(self, timeout=LISTEN_TIMEOUT):
        import vosk, pyaudio
        rec = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=4000)
        stream.start_stream()
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
        return result

if __name__ == '__main__':
    rospy.init_node('voice_test')
    rospy.loginfo("语音测试模式启动")
    
    # 加载语音识别
    recognizer = None
    if os.path.exists(MODEL_DIR):
        recognizer = SpeechRecognizer(MODEL_DIR)
        rospy.loginfo("Vosk 模型已加载")
    else:
        rospy.logerr(f"模型不存在: {MODEL_DIR}")
        exit(1)
    
    while not rospy.is_shutdown():
        input("\n按回车开始语音识别...")
        play_audio("welcome.wav")
        
        text = recognizer.listen_once()
        rospy.loginfo(f"识别结果: {text}")
        
        venue = extract_place(text)
        if venue:
            rospy.loginfo(f"目标场馆: {venue}")
            play_audio(f"confirm_{venue}.wav")
            if venue != "起点":
                rospy.loginfo("开始导航...")
                send_nav_goal(venue)
                play_audio(f"intro_{venue}.wav")
                rospy.loginfo("返回起点...")
                send_nav_goal("起点")
                play_audio("back.wav")
        else:
            rospy.loginfo("未识别到场馆，请重试")
