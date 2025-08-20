import cv2
import numpy as np
import torch
import math
import time
from collections import deque
from src.config import IMAGE_SIZE, TONE_LABELS, PREDICTION_HISTORY_SIZE, MIN_CONFIDENCE_THRESHOLD, TONE_CONFIDENCE_THRESHOLD, TONE_FRAMES_COUNT
from src.model import StabilityDetector

class FrameProcessor:
    def __init__(self, detector, classifier, tone_predictor, stability_detector, text_processor):
        self.detector = detector
        self.classifier = classifier
        self.tone_predictor = tone_predictor
        self.stability_detector = stability_detector
        self.text_processor = text_processor
        
        self.last_detection_time = time.time()
        self.hand_detected_time = None
        self.recognition_started = False
        self.prediction = deque(maxlen=PREDICTION_HISTORY_SIZE)
        self.tone_collection = False
        self.tone_frames = []
        self.tone_start_time = None
        self.after_tone_cooldown = 0
        self.hand_positions = deque(maxlen=10)
        self.movement_threshold = 0.03  # Giảm để nhạy hơn với chuyển động dấu thanh
        self.static_timeout = 0.3
        self.tone_motion_detected_time = None  # Thời điểm phát hiện chuyển động để chờ 0.2s
        self.after_tone_stable_cooldown = 0    # Cooldown sau khi nhận diện dấu thanh
        self.tone_just_processed = False  # Không cho nhận lại dấu thanh liên tục
        self.motion_history = deque(maxlen=15)  # Lưu keypoints các frame gần nhất
        self.motion_energy_history = deque(maxlen=15)
        self.motion_threshold = 0.03  # Ngưỡng biến thiên (tùy chỉnh)
        self.motion_count_required = 7  # Số frame liên tiếp để xác định bắt đầu/kết thúc
        self.gesture_active = False
        self.gesture_start_frame = None
        self.gesture_end_frame = None

    def get_bounding_box(self, landmarks, shape):
        if not landmarks:
            return None
        h, w, _ = shape
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x_min, x_max = min(xs) * w, max(xs) * w
        y_min, y_max = min(ys) * h, max(ys) * h
        return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

    def prepare_image_for_classification(self, image, bbox):
        if bbox is None:
            return None
        x, y, w, h = bbox
        h_img, w_img, _ = image.shape
        x1, y1 = max(0, x - 20), max(0, y - 20)
        x2, y2 = min(w_img, x + w + 20), min(h_img, y + h + 20)
        imgCrop = image[y1:y2, x1:x2]
        if imgCrop.size == 0:
            return None
        imgWhite = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8) * 255
        aspectRatio = h / w
        if aspectRatio > 1:
            k = IMAGE_SIZE / h
            wCal = int(round(k * w))
            imgResize = cv2.resize(imgCrop, (wCal, IMAGE_SIZE))
            imgResize = imgResize[:, :IMAGE_SIZE]
            wGap = (IMAGE_SIZE - imgResize.shape[1]) // 2
            imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
        else:
            k = IMAGE_SIZE / w
            hCal = int(round(k * h))
            imgResize = cv2.resize(imgCrop, (IMAGE_SIZE, hCal))
            imgResize = imgResize[:IMAGE_SIZE, :]
            hGap = (IMAGE_SIZE - imgResize.shape[0]) // 2
            imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize
        return imgWhite

    def is_hand_moving(self, threshold=None):
        if len(self.hand_positions) < 2:
            return False
        arr = np.array(self.hand_positions)
        diffs = np.diff(arr, axis=0)
        total_move = np.sum(np.linalg.norm(diffs, axis=1))
        return total_move > (threshold if threshold is not None else self.movement_threshold)

    def detect_tone_action(self, keypoints):
        if self.tone_collection:
            return False
        self.tone_collection = True
        self.tone_frames = []
        self.tone_start_time = time.time() + 0.2  # Bắt đầu thu thập sau 0.2s
        self.tone_collecting = False
        self.tone_first_kpts = keypoints
        return True

    def finalize_tone_recognition(self):
        frames_needed = TONE_FRAMES_COUNT
        duration = time.time() - self.tone_start_time if self.tone_start_time else 0
        # Chỉ nhận diện nếu tổng thời gian động >= 1.2s
        if (len(self.tone_frames) >= frames_needed or duration >= 1.5or self.tone_collecting is False):
            if duration < 1.2:
                print(f"[INFO] Động tác quá ngắn ({duration:.2f}s), bỏ qua nhận diện dấu thanh.")
                self.reset_tone_state()
                self.after_tone_stable_cooldown = time.time() + 0.5
                self.hand_positions.clear()
                return
            try:
                while len(self.tone_frames) < frames_needed:
                    self.tone_frames.append(self.tone_frames[-1])
                tone, confidence = self.tone_predictor.predict(self.tone_frames[:frames_needed])
                if tone and confidence >= TONE_CONFIDENCE_THRESHOLD:
                    self.text_processor.apply_tone_to_word(tone)
                    print(f"[INFO] Dấu thanh được áp dụng: {tone}, confidence: {confidence:.2f}")
                    self.tone_just_processed = True  # Chỉ chặn nhận diện lại khi thành công
                    self.reset_tone_state()
                    self.after_tone_stable_cooldown = time.time() + 0.7  # Tăng cooldown để ổn định
                    self.hand_positions.clear()
                else:
                    print(f"[INFO] Độ tin cậy thấp: {confidence:.2f}, cho phép nhận diện lại")
                    # Không đặt tone_just_processed = True, cho phép nhận diện lại
                    self.reset_tone_state()
                    self.after_tone_stable_cooldown = time.time() + 0.3  # Cooldown ngắn hơn
                    self.hand_positions.clear()
            except Exception as e:
                print(f"[ERROR] Lỗi khi chạy mô hình LSTM: {e}")
                # Khi có lỗi, cũng cho phép nhận diện lại
                self.reset_tone_state()
                self.after_tone_cooldown = time.time() + 0.3
                self.hand_positions.clear()
        # Nếu chưa đủ điều kiện thì tiếp tục thu thập

    def reset_tone_state(self):
        self.tone_collection = False
        self.tone_frames = []
        self.tone_start_time = None
        self.last_tone_frame_time = None
        self.stability_detector.reset()
        self.after_tone_cooldown = time.time() + 1

    def process_character_recognition(self, processed_image):
        try:
            results, index, _, confidence = self.classifier.prediction(processed_image, draw=False)
            self.prediction.append(index.item())
            probabilities = torch.softmax(torch.tensor(results), dim=0).numpy()
            recent_predictions = list(self.prediction)[-PREDICTION_HISTORY_SIZE:]
            most_common = self.text_processor.most_common_value(recent_predictions)
            if most_common == index.item() and confidence > MIN_CONFIDENCE_THRESHOLD:
                from src.config import CLASSES
                raw_character = CLASSES[index]
                if self.text_processor.process_character(raw_character):
                    self.last_detection_time = time.time()
                    self.tone_just_processed = False  # Cho phép nhận dấu thanh mới sau khi có ký tự mới
                    # Đặt cooldown ngắn để có thời gian cho dấu thanh
                    self.after_tone_cooldown = time.time() + 0.3
                    return True
        except Exception as e:
            print(f"[ERROR] Lỗi trong process_character_recognition: {e}")
        return False

    def reset_hand_state(self):
        self.stability_detector.reset()
        self.reset_tone_state()
        self.hand_positions.clear()
        self.hand_detected_time = None
        self.recognition_started = False
        self.text_processor.just_processed_character = False

    def compute_motion_energy(self, kpts):
        if len(self.motion_history) == 0:
            return 0.0
        prev_kpts = self.motion_history[-1]
        # Tính tổng khoảng cách Euclidean giữa các keypoints
        diffs = np.linalg.norm(kpts - prev_kpts, axis=1)
        return np.mean(diffs)

    def update_motion_state(self, kpts):
        self.motion_history.append(kpts)
        if len(self.motion_history) < 2:
            self.motion_energy_history.append(0.0)
            return

        energy = self.compute_motion_energy(kpts)
        self.motion_energy_history.append(energy)

        # Kiểm tra liên tiếp vượt/ngưỡng
        above = [e > self.motion_threshold for e in list(self.motion_energy_history)[-self.motion_count_required:]]
        below = [e < self.motion_threshold for e in list(self.motion_energy_history)[-self.motion_count_required:]]

        if not self.gesture_active and all(above) and len(above) == self.motion_count_required:
            self.gesture_active = True
            self.gesture_start_frame = len(self.motion_history)
            print("[INFO] Gesture START detected")
        elif self.gesture_active and all(below) and len(below) == self.motion_count_required:
            self.gesture_active = False
            self.gesture_end_frame = len(self.motion_history)
            print("[INFO] Gesture END detected")

    def process_frame(self, frame, no_hand_threshold=1):
        try:
            current_time = time.time()
            hands, image = self.detector.findHands(frame)
            frame_out = frame.copy()
            if current_time < self.after_tone_stable_cooldown:
                if hands:
                    hand = hands[0]
                    if 'landmark' in hand:
                        # Sử dụng cùng logic với phần chính để tính vị trí đại diện
                        pinky_xy = (hand['landmark'][20].x, hand['landmark'][20].y)
                        index_xy = (hand['landmark'][8].x, hand['landmark'][8].y)
                        avg_xy = ((pinky_xy[0] + index_xy[0]) / 2, (pinky_xy[1] + index_xy[1]) / 2)
                        self.hand_positions.append(avg_xy)
                return frame_out
            if hands:
                hand = hands[0]
                if self.hand_detected_time is None:
                    self.hand_detected_time = current_time
                    self.recognition_started = False
                time_elapsed = current_time - self.hand_detected_time
                if not self.recognition_started and time_elapsed >= 0.3:
                    self.recognition_started = True
                if 'landmark' in hand and self.recognition_started:
                    kpts = np.array([[lm.x, lm.y, lm.z] for lm in hand['landmark']])
                    if len(kpts) != 21:
                        print(f"[WARNING] Số lượng landmarks không đúng: {len(kpts)} thay vì 21")
                        kpts = np.pad(kpts, ((0, max(0, 21 - len(kpts))), (0, 0)), mode='constant')[:21]
                    # --- Thêm cập nhật motion energy ---
                    self.update_motion_state(kpts)
                    # --- Kết thúc thêm ---
                    self.stability_detector.add_keypoints(kpts.flatten())
                    wrist = hand['landmark'][0]
                    # Sử dụng vị trí ngón út (20) và ngón trỏ (8) để phát hiện chuyển động dấu thanh
                    pinky_xy = (hand['landmark'][20].x, hand['landmark'][20].y)
                    index_xy = (hand['landmark'][8].x, hand['landmark'][8].y)
                    # Tính trung bình vị trí 2 ngón để có 1 điểm đại diện ổn định
                    avg_xy = ((pinky_xy[0] + index_xy[0]) / 2, (pinky_xy[1] + index_xy[1]) / 2)
                    self.hand_positions.append(avg_xy)
                    hand_is_moving = self.is_hand_moving()
                    can_detect_tone = not self.tone_collection and not self.text_processor.just_processed_character and current_time >= self.after_tone_cooldown and not self.tone_just_processed
                    if can_detect_tone and hand_is_moving:
                        if self.tone_motion_detected_time is None:
                            self.tone_motion_detected_time = current_time
                        elif (current_time - self.tone_motion_detected_time) >= 0.2:
                            self.detect_tone_action(kpts)
                            self.tone_motion_detected_time = None
                    else:
                        self.tone_motion_detected_time = None
                    if self.tone_collection:
                        # Bắt đầu thu thập sau 0.2s kể từ khi phát hiện động
                        if not self.tone_collecting and current_time >= self.tone_start_time:
                            self.tone_collecting = True
                            self.tone_frames.append(self.tone_first_kpts)
                            self.last_tone_frame_time = current_time
                        if self.tone_collecting:
                            # Nếu còn động thì tiếp tục thu thập, nếu tĩnh thì dừng
                            interval = 1.5/ TONE_FRAMES_COUNT
                            if hand_is_moving and len(self.tone_frames) < TONE_FRAMES_COUNT and (current_time - self.last_tone_frame_time) >= interval:
                                self.tone_frames.append(kpts)
                                self.last_tone_frame_time = current_time
                            elif not hand_is_moving:
                                self.tone_collecting = False  # Dừng thu thập nếu tay tĩnh
                                print(f"[INFO] Tay tĩnh, dừng thu thập frame cho dấu thanh")
                            # Kết thúc khi đủ 30 frame, hết 1.2s hoặc tay tĩnh
                            if len(self.tone_frames) >= TONE_FRAMES_COUNT or (current_time - self.tone_start_time) >= 1.5or not self.tone_collecting:
                                self.tone_collecting = False
                                self.finalize_tone_recognition()
                    else:
                        if time.time() >= self.after_tone_cooldown and self.stability_detector.is_stable() and not self.tone_collection:
                            bbox = self.get_bounding_box(hand['landmark'], image.shape)
                            processed_image = self.prepare_image_for_classification(image, bbox)
                            if processed_image is not None:
                                if self.process_character_recognition(processed_image):
                                    self.text_processor.just_processed_character = True
                                else:
                                    self.text_processor.just_processed_character = False
            else:
                self.tone_motion_detected_time = None
                if self.hand_detected_time and (current_time - self.hand_detected_time) >= no_hand_threshold:
                    self.reset_hand_state()
                if self.text_processor.current_word and (current_time - self.last_detection_time) >= no_hand_threshold:
                    self.text_processor.finalize_word()
            return frame_out
        except Exception as e:
            print(f"[ERROR] Lỗi trong process_frame: {e}")
            return frame.copy() 