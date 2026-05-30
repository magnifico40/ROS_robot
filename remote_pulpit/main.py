import sys
import cv2
import numpy as np
import zenoh
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QTextEdit,
    QLineEdit, QWidgetAction, QLabel, QWidget, QPushButton, 
    QHBoxLayout, #widgets horizontally
    QVBoxLayout, #widgets vertically
    QGridLayout,  #widgets in grid
    QFrame,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSlot, QUrl
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from zenoh_handler import ZenohHandler
from config import SERVER_IP, PORT, STOP_INTERVAL, MAN_CMD_VEL_INTERVAL
import os, sys
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-web-security"

from pyqtlet2 import MapWidget, L
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QFileDialog


class RoundButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(200, 200)
        self.setMaximumSize(600, 600)
        self._stopped = False
        self._update_style(600)

    def toggle_stop(self):
        self._stopped = not self._stopped
        size = min(self.width(), self.height())
        if self._stopped:
            self.setText("RESET")
            self._update_style_color(size, "darkred", "#8b0000")
        else:
            self.setText("STOP")
            self._update_style_color(size, "red", "darkred")

    def _update_style(self, size):
        self._update_style_color(size, "red", "darkred")

    def _update_style_color(self, size, color, pressed_color):
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-size: 32px;
                font-weight: bold;
                border-radius: {size // 2}px;
            }}
            QPushButton:pressed {{
                background-color: {pressed_color};
            }}
        """)

    def resizeEvent(self, event):
        size = min(self.width(), self.height())
        color = "darkred" if self._stopped else "red"
        pressed = "#8b0000" if self._stopped else "darkred"
        self._update_style_color(size, color, pressed)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return min(width, 600)


class SteeringWidget(QFrame):
    def __init__(self, zenoh_handler, parent=None):
        super().__init__(parent)
        self.zenoh = zenoh_handler
        self._manual = False  

        self._cmd_timer = QTimer()
        self._cmd_timer.setInterval(MAN_CMD_VEL_INTERVAL)
        self._cmd_timer.timeout.connect(self._send_current_cmd)

        self._linear  = 0.0
        self._angular = 0.0

        self._build_ui()

    def _build_ui(self):
       
        main_layout = QVBoxLayout(self) 

        mode_row = QHBoxLayout()
        self._btn_manual = QPushButton("Manual")
        self._btn_auto   = QPushButton("Auto")

        self._btn_manual.setCheckable(True)
        self._btn_auto.setCheckable(True)
        self._btn_auto.setChecked(True)  

        self._btn_manual.setStyleSheet(self._mode_style())
        self._btn_auto.setStyleSheet(self._mode_style())

        self._btn_manual.clicked.connect(lambda: self._set_mode("manual"))
        self._btn_auto.clicked.connect(lambda: self._set_mode("auto"))

        mode_row.addWidget(self._btn_manual)
        mode_row.addWidget(self._btn_auto)
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        grid = QGridLayout()

        self._btn_forward  = self._make_arrow_btn("▲")
        self._btn_backward = self._make_arrow_btn("▼")
        self._btn_left     = self._make_arrow_btn("◄")
        self._btn_right    = self._make_arrow_btn("►")

        grid.addWidget(self._btn_forward,  0, 1) #btn, row, col
        grid.addWidget(self._btn_left,     1, 0)
        grid.addWidget(self._btn_right,    1, 2)
        grid.addWidget(self._btn_backward, 2, 1)

        main_layout.addStretch()
        main_layout.addLayout(grid)
        main_layout.addStretch()

        self._set_mode("auto") 

    def _make_arrow_btn(self, text):
        btn = QPushButton(text)
        btn.setFixedSize(80, 80)
        btn.setStyleSheet("""
            QPushButton {
                background: #313244;
                color: white;
                font-size: 28px;
                border-radius: 8px;
            }
            QPushButton:pressed { background: #89b4fa; }
            QPushButton:disabled { background: #1a1a1a; color: #444; }
        """)

        btn.pressed.connect(lambda t=text: self._on_pressed(t))
        btn.released.connect(self._on_released)
        return btn

    def _highlight_btn(self, btn, active: bool):
        size = btn.size()
        if active:
            btn.setStyleSheet("""
                QPushButton {
                    background: #89b4fa;
                    color: white;
                    font-size: 28px;
                    border-radius: 8px;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background: #313244;
                    color: white;
                    font-size: 28px;
                    border-radius: 8px;
                }
                QPushButton:pressed { background: #89b4fa; }
                QPushButton:disabled { background: #1a1a1a; color: #444; }
            """)
    def _on_pressed(self, direction):
        if not self._manual:
            return
        speeds = {
            "▲": ( 0.5,  0.0),
            "▼": (-0.5,  0.0),
            "◄": ( 0.0,  0.5),
            "►": ( 0.0, -0.5),
        }
        self._linear, self._angular = speeds.get(direction, (0.0, 0.0)) #(0.0, 0.0) - default val
        self._cmd_timer.start()

    def _on_released(self):
        self._cmd_timer.stop()
        self._linear  = 0.0
        self._angular = 0.0
        self.zenoh.send_cmd_vel(0.0, 0.0)

    def _send_current_cmd(self):
        self.zenoh.send_cmd_vel(self._linear, self._angular)

    def _set_mode(self, mode: str):
        self._manual = True if mode =="manual" else False

        self._btn_manual.setChecked(self._manual)
        self._btn_auto.setChecked(not self._manual)

        self._btn_forward.setEnabled(self._manual)
        self._btn_backward.setEnabled(self._manual)
        self._btn_left.setEnabled(self._manual)
        self._btn_right.setEnabled(self._manual)

        if not self._manual:
            self._cmd_timer.stop()
            self.zenoh.send_cmd_vel(0.0, 0.0)

        self.zenoh.send_steering_mode(mode)

    def _mode_style(self):
        return """
            QPushButton {
                background: #313244;
                color: white;
                border-radius: 6px;
                padding: 6px 16px;
            }
            QPushButton:checked { background: #89b4fa; color: black; }
        """

    def key_press(self, key):
        if not self._manual:
            return
        mapping = {
            Qt.Key.Key_W: ("▲", self._btn_forward),
            Qt.Key.Key_S: ("▼", self._btn_backward),
            Qt.Key.Key_A: ("◄", self._btn_left),
            Qt.Key.Key_D: ("►", self._btn_right),
        }
        if key in mapping:
            direction, btn = mapping[key]
            self._highlight_btn(btn, True) 
            self._on_pressed(direction)

    def key_release(self, key):
        mapping = {
            Qt.Key.Key_W: self._btn_forward,
            Qt.Key.Key_S: self._btn_backward,
            Qt.Key.Key_A: self._btn_left,
            Qt.Key.Key_D: self._btn_right,
        }
        if key in mapping:
            self._highlight_btn(mapping[key], False)  
            self._on_released()

class MapApp(QWidget):
    def __init__(self, zenoh):
        super().__init__()
        self.zenoh = zenoh
        self.markers = []
        self.leaflet_markers = []
        self.mode = "fence"

        layout = QVBoxLayout(self)

        btns = QHBoxLayout()
        btn_clear = QPushButton("Clean All")
        btn_undo = QPushButton("Undo")
        btn_export = QPushButton("Send map to robot")
        self.btn_path = QPushButton("Path")
        self.btn_fence = QPushButton("Fence")

        self.btn_path.setCheckable(True)
        self.btn_fence.setCheckable(True)
        self.btn_fence.setChecked(True)

        btn_clear.clicked.connect(self.clear)
        btn_undo.clicked.connect(self.undo)
        btn_export.clicked.connect(self.export)
        self.btn_path.clicked.connect(self.set_path)
        self.btn_fence.clicked.connect(self.set_fence)

        btns.addWidget(btn_clear)
        btns.addWidget(btn_undo)
        btns.addWidget(btn_export)
        btns.addWidget(self.btn_path)
        btns.addWidget(self.btn_fence)

        self.map_widget = MapWidget()
        self.map = L.map(self.map_widget)
        self.map.setView([54.371949, 18.616702], 13)
        L.tileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            {"maxZoom": 19}
        ).addTo(self.map)
        self.map.clicked.connect(self.on_click)

        layout.addLayout(btns)
        layout.addWidget(self.map_widget)


    def set_path(self):
        self.mode = "path"
        self.btn_fence.setChecked(False)
        self.btn_path.setChecked(True)

    def set_fence(self):
        self.mode = "fence"
        self.btn_fence.setChecked(True)
        self.btn_path.setChecked(False)

    def on_click(self, event):
        lat, lng = event["latlng"]["lat"], event["latlng"]["lng"]
        n = len(self.markers) + 1
        m = L.marker([lat, lng])
        m.bindTooltip(f"{n}")
        m.addTo(self.map)
        self.leaflet_markers.append(m)
        self.markers.append({"lat": lat, "lng": lng})
        self.update_line()

    def update_line(self):
        if hasattr(self, "polyline"):
            self.map.removeLayer(self.polyline)
        if len(self.markers) < 2:
            return
        coords = [[m["lat"], m["lng"]] for m in self.markers]
        if len(self.markers) >= 3 and self.mode == "fence":
            coords.append(coords[0])
        self.polyline = L.polyline(coords, {"color": "blue", "weight": 2})
        self.polyline.addTo(self.map)

    def undo(self):
        if self.leaflet_markers:
            self.map.removeLayer(self.leaflet_markers.pop())
            self.markers.pop()
            self.update_line()

    def clear(self):
        for m in self.leaflet_markers:
            self.map.removeLayer(m)
        self.leaflet_markers.clear()
        self.markers.clear()
        if hasattr(self, "polyline"):
            self.map.removeLayer(self.polyline)
            del self.polyline

    def export(self):
        
        data = self.markers.copy()
        if self.mode == "fence" and len(data) >= 3:
            data.append(data[0])
        print("n")
        if data is not None:
            print("e")
            self.zenoh.send_map_data(data)
        #path, _ = QFileDialog.getSaveFileName(self, "Zapisz", "znaczniki.json", "JSON (*.json)")
        #if path:
        #    with open(path, "w") as f:
        #        json.dump(data, f, indent=2)

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pulpit")

        self.zenoh = ZenohHandler() 

        self.organise_UI()

        
        self.zenoh.pqt_sig_status.connect(self._on_status) #status signal for info
        self.zenoh.pqt_sig_frame.connect(self._on_frame) #video frame

    def organise_UI(self):
        def _organise_main_window():
            main_splitter = QSplitter(Qt.Orientation.Vertical)
            top_splitter = QSplitter(Qt.Orientation.Horizontal)
            bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

            #camera
            self.camera_widget = QLabel("No Signal")
            self.camera_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.camera_widget.setSizePolicy(
                QSizePolicy.Policy.Ignored,   
                QSizePolicy.Policy.Ignored    
            )

            #steering
            self.steering_widget = SteeringWidget(self.zenoh)

            #Map
            self.map_widget = MapApp(self.zenoh)
            
            #Stop
            self.stop_widget = QWidget()
            stop_layout = QVBoxLayout(self.stop_widget)
            self.stop_button = RoundButton("STOP")
           
            
            self.stop_button.clicked.connect(self.send_stop)
            stop_layout.addWidget(self.stop_button, alignment=Qt.AlignmentFlag.AlignCenter)
           

            self.camera_widget.setStyleSheet("border: 2px solid #89b4fa;")
            self.steering_widget.setStyleSheet("border: 2px solid #89b4fa;")
            self.stop_widget.setStyleSheet("border: 2px solid #89b4fa;")

            top_splitter.addWidget(self.camera_widget)
            top_splitter.addWidget(self.steering_widget)
            top_splitter.addWidget(self.stop_widget)
            bottom_splitter.addWidget(self.map_widget)
            #bottom_splitter.addWidget(self.stop_widget)
            main_splitter.addWidget(top_splitter)
            main_splitter.addWidget(bottom_splitter)
            self.setCentralWidget(main_splitter)

        def _organise_log_widget():
            self.log_widget = QLabel("Status: Disconnected")

        def _organise_menu():
            corner_container = QWidget()
            corner_layout = QHBoxLayout(corner_container)
            corner_layout.setContentsMargins(0, 0, 10, 0)
            corner_layout.addWidget(self.log_widget)
            self.menuBar().setCornerWidget(corner_container, Qt.Corner.TopRightCorner)

            serv_menu = self.menuBar().addMenu("Server Options")
            ip_serv_submenu = serv_menu.addMenu("Server")

            self.ip_input = QLineEdit()
            self.ip_input.setPlaceholderText("Server IP: ")
            self.ip_input.setText("16.171.226.101")
            self.ip_input.setFixedWidth(150)

            ip_widget_action = QWidgetAction(self)
            ip_widget_action.setDefaultWidget(self.ip_input)
            ip_serv_submenu.addAction(ip_widget_action)
            confirm_ip_action = ip_serv_submenu.addAction("OK")

            connect_action    = serv_menu.addAction("Connect")
            disconnect_action = serv_menu.addAction("Disconnect")

            connect_action.triggered.connect(self.connect_to_server)
            disconnect_action.triggered.connect(self.disconnect_from_server)
            confirm_ip_action.triggered.connect(self.confirm_ip)

        _organise_main_window()
        _organise_log_widget()
        _organise_menu()

    def confirm_ip(self):
        self.zenoh.server_ip = self.ip_input.text().strip()
        self.log_widget.setText(f"IP: {self.zenoh.server_ip}")

    def connect_to_server(self):
        self.zenoh.server_ip = self.ip_input.text().strip()
        self.zenoh.connect()   

    def disconnect_from_server(self):
        self.zenoh.disconnect()
        self.camera_widget.setPixmap(QPixmap()) 
        self.camera_widget.setText("Disconnected")

    def _on_status(self, text: str):
        self.log_widget.setText(text)

    def _on_frame(self, frame: np.ndarray):
        if not self.zenoh.is_connected():  
            return
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.camera_widget.setPixmap(
            QPixmap.fromImage(qt_img).scaled(
                self.camera_widget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            )
        
    def send_stop(self):
        self.stop_button.toggle_stop()
        if self.stop_button._stopped:
            self.zenoh.send_stop()       
        else:
            self.zenoh.reset_stop()

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        self.steering_widget.key_press(event.key())

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat(): 
            return
        self.steering_widget.key_release(event.key())     





if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.showMaximized()
    window.show()
    sys.exit(app.exec())