import numpy as np
from collections import deque


# Stability detector for steady hand detection
class StabilityDetector:
    def __init__(self, max_frames=30, stability_threshold=0.02):
        self.max_frames = max_frames
        self.stability_threshold = stability_threshold
        self.keypoints_history = deque(maxlen=max_frames)
    
    def add_keypoints(self, keypoints):
        self.keypoints_history.append(keypoints)
    
    def is_stable(self):
        if len(self.keypoints_history) < self.max_frames:
            return False
        arr = np.array(self.keypoints_history)
        var = np.var(arr, axis=0)
        return np.mean(var) < self.stability_threshold
    
    def reset(self):
        self.keypoints_history.clear()

