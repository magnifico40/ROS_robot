import zenoh
import cv2
import time
import numpy as np
import json
import struct
import threading
from PyQt6.QtCore import QObject, pyqtSignal
from config import SERVER_IP, PORT, STOP_INTERVAL, MAN_CMD_VEL_INTERVAL


class ZenohHandler(QObject):
    pqt_sig_frame          = pyqtSignal(np.ndarray)
    pqt_sig_status         = pyqtSignal(str)
    pqt_sig_connection_lost = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.server_ip  = SERVER_IP
        self.tcp_port   = PORT
        
        #1 thread for GUI, 1 for zenoh
        self._lock       = threading.Lock()
        #protected by lock, so both thread dont modify at the same time
        self.session_tcp = None
        self._publishers = {}  

        self._show_video = False
        self._robot_active = True
        self._heartbeat_timer = None

    def connect(self):
        threading.Thread(target=self._connect_worker, daemon=True).start()

    def _connect_worker(self):
        try:
            self._show_video = True
            self.pqt_sig_status.emit("Connecting to server...")
            conf_tcp = zenoh.Config()
            conf_tcp.insert_json5("connect/endpoints", f'["tcp/{self.server_ip}:{self.tcp_port}"]')
            self.session_tcp = zenoh.open(conf_tcp)

            #subscribers
            self._sub_camera = self.session_tcp.declare_subscriber("camera/image_raw/compressed",self._on_camera_frame)

            #publishers
            self._publishers["waypoints"] = self.session_tcp.declare_publisher("rt/waypoints")
            self._publishers["robot_status"] = self.session_tcp.declare_publisher("rt/robot_status")
            self._publishers["manual_cmd_vel"]   = self.session_tcp.declare_publisher("rt/manual_cmd_vel")
            self._publishers["steering_mode"] = self.session_tcp.declare_publisher("rt/steering_mode")
            self._publishers["map_data"] = self.session_tcp.declare_publisher("rt/map_data")
            self.start_heartbeat()

            self.pqt_sig_status.emit("Connected")

        except Exception as e:
            self.pqt_sig_status.emit(f"Error: {e}")

    def disconnect(self):
        self._show_video = False
        self._robot_active = False
        with self._lock: #only 1 thread modifies self.session and publishers
            if self.session_tcp:        
                self.session_tcp.close()
                self.session_tcp = None
                self._publishers.clear()
        self.pqt_sig_status.emit("Disconnected")

    def is_connected(self) -> bool:
        return self.session_tcp is not None  

    def _on_camera_frame(self, sample):
        if not self._show_video: 
            return
        payload = bytes(sample.payload)
        idx = payload.find(b'\xff\xd8\xff')
        if idx == -1:
            return
        print("data in!")
        arr = np.frombuffer(payload[idx:], np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            self.pqt_sig_frame.emit(img) 
    
    def _publish(self, key: str, payload: bytes):
        with self._lock:
            pub = self._publishers.get(key)  
            if pub:
                pub.put(payload)
    
    def start_heartbeat(self):
        self._heartbeat_timer = threading.Timer(STOP_INTERVAL, self._heartbeat_worker) #new thread: publishes robot_status every 100ms
        self._heartbeat_timer.daemon = True
        self._heartbeat_timer.start()
    
    def _heartbeat_worker(self):
        while self.is_connected():
            payload = struct.pack('<4s?', b'\x00\x01\x00\x00', self._robot_active)  
            self._publish("robot_status", payload)
            time.sleep(STOP_INTERVAL)

    def send_stop(self):
        self._robot_active = False
        payload = struct.pack('<4s?', b'\x00\x01\x00\x00', False)
        self._publish("robot_status", payload)
        self.pqt_sig_status.emit("STOP")

    def reset_stop(self):
        self._robot_active = True
        payload = struct.pack('<4s?', b'\x00\x01\x00\x00', True)
        self._publish("robot_status", payload)
        self.pqt_sig_status.emit("Reset Stop")

    def send_steering_mode(self, mode: str):  #manual/auto
        payload = b'\x00\x01\x00\x00'
        text = mode.encode('utf-8') + b'\x00'
        payload += struct.pack('<I', len(text)) + text
        self._publish("steering_mode", payload)

    def send_cmd_vel(self, linear: float, angular: float):
        payload = b'\x00\x01\x00\x00'
        payload += struct.pack('<6d', linear, 0.0, 0.0, 0.0, 0.0, angular)
        self._publish("manual_cmd_vel", payload)
    
    def send_map_data(self, waypoints:list):
        print("t")
        #if not self.is_connected():
        #    self.pqt_sig_status.emit("Not connected")
        #    return
        
        data = json.dumps({"waypoints": waypoints})
        payload = b'\x00\x01\x00\x00'
        text = data.encode('utf-8') + b'\x00'
        payload += struct.pack('<I', len(text)) + text
        self._publish("waypoints", payload)
        print("sent!")


