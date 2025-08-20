import os
import sys
import time
import cv2
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QPushButton, QTextEdit, QFrame, 
                            QGridLayout, QProgressBar, QGroupBox, QSplitter, QSizePolicy)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

from src.model import StabilityDetector
from src.text_processor import TextProcessor
from src.frame_processor import FrameProcessor
from src.hand_tracking import handDetector
from src.classification import Classifier
from src.tone_predictor import TonePredictor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VideoThread(QThread):
    """Thread để xử lý video"""
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(dict)
    
    def __init__(self, frame_processor):
        super().__init__()
        self.frame_processor = frame_processor
        self.cap = None
        self.is_running = False
        self.fps = 0
        self.pTime = 0
        
    def start_camera(self):
        """Bắt đầu camera"""
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Tăng kích thước camera
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.is_running = True
        self.start()
        
    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.quit()
        self.wait()
        
    def run(self):
        """Vòng lặp chính của thread"""
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Xử lý frame
                frame_out = self.frame_processor.process_frame(frame)
                
                # Tính FPS
                cTime = time.time()
                self.fps = 1 / (cTime - self.pTime) if self.pTime else 0
                self.pTime = cTime
                
                # Emit frame
                self.frame_ready.emit(frame_out)
                
                # Emit status
                status_data = {
                    'fps': self.fps,
                    'status': "Nhận dấu thanh" if self.frame_processor.tone_collection else "Nhận ký tự",
                    'tone_collection': self.frame_processor.tone_collection,
                    'current_char': self.frame_processor.text_processor.current_word[-1] if self.frame_processor.text_processor.current_word else "",
                    'tone_prediction': self.frame_processor.tone_predictor.current_prediction or "",
                    'tone_confidence': self.frame_processor.tone_predictor.current_confidence or 0,
                    'display_text': self.frame_processor.text_processor.get_display_text() or "",
                    'prediction_threshold': self.frame_processor.tone_predictor.prediction_threshold
                }
                self.status_update.emit(status_data)
                
            self.msleep(50)  # ~20 FPS

class ModernSignLanguageQt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nhận Diện Ngôn Ngữ Ký Hiệu - PyQt5")
        self.setGeometry(100, 100, 1600, 900)  # Tăng kích thước cửa sổ chính
        
        # Thiết lập theme
        self.setup_theme()
        
        # Khởi tạo AI components
        self.detector = handDetector(maxHands=1)
        self.classifier = Classifier()
        self.tone_predictor = TonePredictor(model_type="lstm")
        self.stability_detector = StabilityDetector(max_frames=12, stability_threshold=0.025)
        self.text_processor = TextProcessor()
        self.frame_processor = FrameProcessor(
            detector=self.detector,
            classifier=self.classifier,
            tone_predictor=self.tone_predictor,
            stability_detector=self.stability_detector,
            text_processor=self.text_processor
        )
        
        # Khởi tạo video thread
        self.video_thread = VideoThread(self.frame_processor)
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.status_update.connect(self.update_status)
        
        # Tạo giao diện
        self.init_ui()
        
    def setup_theme(self):
        """Thiết lập theme hiện đại"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font: bold 16px 'Segoe UI';
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                color: white;
                padding: 12px 24px;
                font: bold 16px 'Segoe UI';
                border-radius: 6px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton#startBtn {
                background-color: #27ae60;
            }
            QPushButton#startBtn:hover {
                background-color: #229954;
            }
            QPushButton#stopBtn {
                background-color: #e74c3c;
            }
            QPushButton#stopBtn:hover {
                background-color: #c0392b;
            }
            QPushButton#clearBtn {
                background-color: #f39c12;
            }
            QPushButton#clearBtn:hover {
                background-color: #e67e22;
            }
            QLabel#statusLabel {
                font: bold 16px 'Segoe UI';
                color: #2c3e50;
                padding: 8px;
                border-radius: 4px;
                background-color: #ecf0f1;
            }
            QLabel#valueLabel {
                font: bold 14px 'Segoe UI';
                color: #3498db;
                padding: 4px;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                padding: 8px;
                font-family: 'Segoe UI';
                font-size: 20px;
                font-weight: bold;
                background-color: #f8f9fa;
            }
        """)
        
    def init_ui(self):
        """Khởi tạo giao diện"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout chính
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        self.create_header(main_layout)
        
        # Content area với splitter
        content_splitter = QSplitter(Qt.Horizontal)
        content_splitter.setChildrenCollapsible(False)  # Ngăn panel thu nhỏ quá mức
        main_layout.addWidget(content_splitter)
        
        # Video panel
        video_widget = self.create_video_panel()
        content_splitter.addWidget(video_widget)
        
        # Info panel
        info_widget = self.create_info_panel()
        info_widget.setMinimumWidth(400)  # Đặt chiều rộng tối thiểu cố định
        info_widget.setMaximumWidth(400)  # Đặt chiều rộng tối đa cố định
        content_splitter.addWidget(info_widget)
        
        # Thiết lập tỷ lệ
        content_splitter.setStretchFactor(0, 3)  # Video panel chiếm nhiều không gian hơn
        content_splitter.setStretchFactor(1, 1)
        
    def create_header(self, parent_layout):
        """Tạo header"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3498db, stop:1 #2980b9);
                border-radius: 12px;
                min-height: 80px;
                max-height: 80px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        title_label = QLabel("🤟 NHẬN DIỆN NGÔN NGỮ KÝ HIỆU")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            color: white;
            font: bold 24px 'Segoe UI';
            background: transparent;
        """)
        
        header_layout.addWidget(title_label)
        parent_layout.addWidget(header_frame)
        
    def create_video_panel(self):
        """Tạo panel video"""
        video_group = QGroupBox("📹 Camera Nhận Diện")
        video_layout = QVBoxLayout(video_group)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)  # Tăng kích thước video
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #bdc3c7;
                border-radius: 8px;
                background-color: #f8f9fa;
                color: #6c757d;
                font-size: 16px;
            }
        """)
        self.video_label.setText("Nhấn 'Bắt đầu' để khởi động camera")
        
        video_layout.addWidget(self.video_label)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Bắt đầu")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_camera)
        
        self.stop_btn = QPushButton("⏹ Dừng lại")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_camera)
        
        self.clear_btn = QPushButton("🗑️ Xóa tất cả")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_all_text)
        
        controls_layout.addWidget(self.start_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addWidget(self.clear_btn)
        controls_layout.addStretch()
        
        video_layout.addLayout(controls_layout)
        
        return video_group
        
    def create_info_panel(self):
        """Tạo panel thông tin"""
        info_group = QGroupBox("ℹ️ Thông Tin Hệ Thống")
        info_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)  # Cố định chiều rộng
        info_layout = QVBoxLayout(info_group)
        
        # Status grid
        grid_layout = QGridLayout()
        
        # FPS
        grid_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setObjectName("valueLabel")
        grid_layout.addWidget(self.fps_label, 0, 1)
        
        # Status
        grid_layout.addWidget(QLabel("Trạng thái:"), 1, 0)
        self.status_label = QLabel("Chưa khởi động")
        self.status_label.setObjectName("statusLabel")
        grid_layout.addWidget(self.status_label, 1, 1)
        
        # Current character
        grid_layout.addWidget(QLabel("Ký tự hiện tại:"), 2, 0)
        self.char_label = QLabel("-")
        self.char_label.setObjectName("valueLabel")
        grid_layout.addWidget(self.char_label, 2, 1)
        
        # Tone prediction
        grid_layout.addWidget(QLabel("Dấu thanh:"), 3, 0)
        self.tone_label = QLabel("-")
        self.tone_label.setObjectName("valueLabel")
        grid_layout.addWidget(self.tone_label, 3, 1)
        
        # Confidence
        grid_layout.addWidget(QLabel("Độ tin cậy:"), 4, 0)
        self.confidence_label = QLabel("0.00")
        self.confidence_label.setObjectName("valueLabel")
        grid_layout.addWidget(self.confidence_label, 4, 1)
        
        info_layout.addLayout(grid_layout)
        
        # Text display
        text_label = QLabel("Văn bản nhận diện:")
        text_label.setStyleSheet("font: bold 18px 'Segoe UI'; color: #2c3e50; margin-top: 20px;")
        info_layout.addWidget(text_label)
        
        self.text_display = QTextEdit()
        self.text_display.setMinimumHeight(200)  # Tăng chiều cao text display
        self.text_display.setPlainText("Văn bản sẽ hiển thị ở đây...")
        self.text_display.setReadOnly(True)
        info_layout.addWidget(self.text_display)
        
        return info_group
        
    def start_camera(self):
        """Bắt đầu camera"""
        try:
            self.video_thread.start_camera()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("Đang khởi động...")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #d5f4e6;
                    color: #27ae60;
                    font: bold 16px 'Segoe UI';
                    padding: 8px;
                    border-radius: 4px;
                }
            """)
        except Exception as e:
            self.show_error(f"Không thể khởi động camera: {str(e)}")
            
    def clear_all_text(self):
        """Xóa tất cả văn bản đã nhận diện"""
        try:
            # Reset text processor
            self.frame_processor.text_processor.clear_text()
            
            # Clear display
            self.text_display.setPlainText("Văn bản sẽ hiển thị ở đây...")
            
            # Reset các label hiển thị
            self.char_label.setText("-")
            self.tone_label.setText("-")
            self.confidence_label.setText("0.00")
            
            print("[INFO] Đã xóa tất cả văn bản")
            
        except Exception as e:
            print(f"[ERROR] Lỗi khi xóa văn bản: {e}")
            self.show_error(f"Không thể xóa văn bản: {str(e)}")
            
    def stop_camera(self):
        """Dừng camera"""
        self.video_thread.stop_camera()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Đã dừng")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #fadbd8;
                color: #e74c3c;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                border-radius: 4px;
                }
            """)
        self.video_label.setText("Nhấn 'Bắt đầu' để khởi động camera")
        self.video_label.setPixmap(QPixmap())
        
    def update_frame(self, frame):
        """Cập nhật frame video"""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(960, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
    def update_status(self, status_data):
        """Cập nhật thông tin trạng thái"""
        # FPS
        self.fps_label.setText(f"{status_data['fps']:.1f}")
        
        # Status
        status = status_data['status']
        if status_data['tone_collection']:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #fef9e7;
                    color: #e67e22;
                    font: bold 16px 'Segoe UI';
                    padding: 8px;
                    border-radius: 4px;
                }
            """)
        else:
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #d5f4e6;
                    color: #27ae60;
                    font: bold 16px 'Segoe UI';
                    padding: 8px;
                    border-radius: 4px;
                }
            """)
        self.status_label.setText(status)
        
        # Character
        self.char_label.setText(status_data['current_char'] or "-")
        
        # Tone
        self.tone_label.setText(status_data['tone_prediction'] or "-")
        
        # Confidence
        confidence = status_data['tone_confidence']
        self.confidence_label.setText(f"{confidence:.2f}")
        
        if confidence > status_data['prediction_threshold']:
            self.confidence_label.setStyleSheet("color: #27ae60; font: bold 14px 'Segoe UI';")
        elif confidence > 0.5:
            self.confidence_label.setStyleSheet("color: #f39c12; font: bold 14px 'Segoe UI';")
        else:
            self.confidence_label.setStyleSheet("color: #e74c3c; font: bold 14px 'Segoe UI';")
        
        # Text
        if status_data['display_text']:
            self.text_display.setPlainText(status_data['display_text'])
        
    def show_error(self, message):
        """Hiển thị lỗi"""
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Lỗi", message)
        
    def closeEvent(self, event):
        """Xử lý khi đóng ứng dụng"""
        if self.video_thread.is_running:
            self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Segoe UI', 10))  # Thiết lập font mặc định
    window = ModernSignLanguageQt()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()