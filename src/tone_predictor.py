import os
import numpy as np
import tensorflow as tf
import pickle
from src.config import TONE_FRAMES_COUNT, TONE_CONFIDENCE_THRESHOLD

class TonePredictor:
    def __init__(self, model_path=None):
        """Khởi tạo TonePredictor với mô hình LSTM.
        
        Args:
            model_path (str): Đường dẫn đến mô hình LSTM. Nếu None, sẽ sử dụng đường dẫn mặc định
        """
        self.model = None
        self.sequence_length = TONE_FRAMES_COUNT  # 30 frame
        self.prediction_threshold = TONE_CONFIDENCE_THRESHOLD  # 0.8
        
        # Tự động chọn đường dẫn mô hình nếu không được cung cấp
        if model_path is None:
            self.model_path = "trained_models/lstm_model_final.h5"
        else:
            self.model_path = model_path
            
        self.label_encoder = None
        self.classes = []  # Danh sách nhãn sau khi decode từ label_encoder
        self.current_prediction = None
        self.current_confidence = 0.0

        # Tải mô hình và label encoder
        self.load_model()

    def load_model(self):
        """Tải mô hình LSTM và label encoder từ đường dẫn được chỉ định."""
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Không tìm thấy mô hình tại {self.model_path}!")
            return

        try:
            print(f"[INFO] Đang tải mô hình LSTM từ: {self.model_path}")
            
            # Thử tải mô hình với custom_objects để xử lý lỗi tương thích
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                print(f"[WARNING] Lỗi khi tải mô hình thông thường: {e}")
                print("[INFO] Thử tải với custom_objects...")
                
                # Tạo custom_objects để xử lý lỗi batch_shape
                def custom_input_layer(*args, **kwargs):
                    # Loại bỏ batch_shape nếu có
                    if 'batch_shape' in kwargs:
                        del kwargs['batch_shape']
                    return tf.keras.layers.InputLayer(*args, **kwargs)
                
                custom_objects = {
                    'InputLayer': custom_input_layer
                }
                
                try:
                    self.model = tf.keras.models.load_model(
                        self.model_path, 
                        custom_objects=custom_objects,
                        compile=False
                    )
                    print("[INFO] Tải mô hình thành công với custom_objects!")
                except Exception as e2:
                    print(f"[ERROR] Vẫn không thể tải mô hình: {e2}")
                    # Thử tải với compile=False
                    try:
                        self.model = tf.keras.models.load_model(
                            self.model_path, 
                            compile=False
                        )
                        print("[INFO] Tải mô hình thành công với compile=False!")
                    except Exception as e3:
                        print(f"[ERROR] Không thể tải mô hình: {e3}")
                        self.model = None
                        return
            
            # In thông tin chi tiết về mô hình
            if self.model is not None:
                print(f"[INFO] Mô hình LSTM đã tải thành công!")
                print(f"[INFO] Input shape: {self.model.input_shape}")
                print(f"[INFO] Output shape: {self.model.output_shape}")
            
            # Tải label encoder
            encoder_path = self.model_path.replace("_final.h5", "_label_encoder.pkl")
            
            if os.path.exists(encoder_path):
                with open(encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                    self.classes = list(self.label_encoder.classes_)
                    print(f"[INFO] Label encoder đã tải: {self.classes}")
            else:
                print(f"[WARNING] Không tìm thấy label encoder tại {encoder_path}")
                self.label_encoder = None
                self.classes = []

            print(f"[INFO] Số lớp dự đoán: {len(self.classes)}")
        except Exception as e:
            print(f"[ERROR] Lỗi khi tải mô hình hoặc label encoder: {e}")
            self.model = None

    def preprocess_keypoints(self, keypoints_sequence):
        """Chuẩn hóa chuỗi keypoints cho dự đoán LSTM."""
        keypoints_array = np.array(keypoints_sequence)
        
        # Đảm bảo có đủ số frame cần thiết
        if keypoints_array.shape[0] < self.sequence_length:
            padding = np.zeros((self.sequence_length - keypoints_array.shape[0], keypoints_array.shape[1]))
            keypoints_array = np.vstack([keypoints_array, padding])
        else:
            keypoints_array = keypoints_array[:self.sequence_length]
        
        # Reshape cho LSTM: (batch, sequence, features)
        return keypoints_array.reshape(1, self.sequence_length, -1)

    def predict(self, keypoints_sequence):
        """Dự đoán dấu thanh từ chuỗi keypoints."""
        if self.model is None:
            print(f"[WARNING] Không thể dự đoán: Mô hình LSTM chưa được tải.")
            return None, 0.0

        if len(keypoints_sequence) != self.sequence_length:
            print(f"[WARNING] Số frame không đúng: {len(keypoints_sequence)}/{self.sequence_length}")
            return None, 0.0

        try:
            X = self.preprocess_keypoints(keypoints_sequence)
            
            predictions = self.model.predict(X, verbose=0)[0]
            predicted_idx = np.argmax(predictions)
            confidence = predictions[predicted_idx]

            if self.label_encoder:
                predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
            else:
                predicted_label = str(predicted_idx)  # fallback nếu thiếu label_encoder

            self.current_prediction = predicted_label
            self.current_confidence = confidence
            
            return self.current_prediction, self.current_confidence
        except Exception as e:
            print(f"[ERROR] Lỗi khi dự đoán dấu thanh với mô hình LSTM: {e}")
            return None, 0.0
