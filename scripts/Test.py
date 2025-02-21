import cv2
import numpy as np
import tensorflow as tf
import h5py
from collections import deque
import os
import threading

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceRecognizer:
    def __init__(self, emb_file='embeddings.h5', match_thresh=0.795):
        # Initialize face detector and recognition model
        self.detector = cv2.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000_fp16.caffemodel")
        self.face_net = tf.keras.models.load_model(
            'models/mobile_facenet.h5',
            custom_objects={'l2_normalize': lambda x: tf.math.l2_normalize(x, axis=-1)},
            compile=False
        )
        
        # Warmup model
        self.face_net.predict(np.zeros((1, 160, 160, 3)))
        
        # Load embeddings
        with h5py.File(emb_file, 'r') as hf:
            self.embeddings = hf['embeddings'][:]
            self.names = [os.path.splitext(os.path.basename(p.decode()))[0] 
                        for p in hf['paths'][:]]
            
        # Recognition parameters
        self.match_thresh = match_thresh
        self.detection_interval = 5  # Run detection every 5 frames
        self.frame_counter = 0
        self.identity_buffer = deque(maxlen=10)
        
        # Tracking and threading
        self.tracker = cv2.legacy.TrackerMOSSE_create() 
        self.current_roi = None
        self.lock = threading.Lock()
        self.last_result = ("Initializing...", 0)
        self.processing_queue = []
        self.detection_thresh = 0.9
        self.tracking_failure_count = 0

    def detect_faces(self, frame):
        (h, w) = frame.shape[:2]
    # Correct mean values (104, 177, 123) instead of (104, 117, 123)
        blob = cv2.dnn.blobFromImage(
        frame, 
        scalefactor=1.0, 
        size=(300, 300), 
        mean=(104, 177, 123)  # Fixed mean values
    )
        self.detector.setInput(blob)
        detections = self.detector.forward()
    # Rest of the method remains the same...
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.detection_thresh:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                faces.append((x1, y1, x2-x1, y2-y1, confidence))
        return faces

    def process_face(self, frame):
        # Async face processing
        try:
            resized = cv2.resize(frame, (160, 160))
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            resized = (resized.astype('float32') - 127.5) / 128.0
            embedding = self.face_net.predict(resized[None, ...], verbose=0)[0]
            similarities = np.dot(self.embeddings, embedding)
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]
            
            with self.lock:
                if max_sim > self.match_thresh:
                    self.identity_buffer.append(max_idx)
                else:
                    self.identity_buffer.append(-1)

                if self.identity_buffer:
                    counts = {}
                    for idx in self.identity_buffer:
                        if idx != -1:
                            counts[idx] = counts.get(idx, 0) + 1
                    if counts:
                        final_idx = max(counts, key=counts.get)
                        confidence = int(similarities[final_idx] * 100)
                        self.last_result = (self.names[final_idx], confidence)
                    else:
                        self.last_result = ("Unknown", 0)
        except Exception as e:
            print(f"Processing error: {e}")

    def process_frame(self, frame):
        self.frame_counter += 1
        updated = False
        face_rect = None
        
        # Tracking logic
        if self.current_roi and self.tracking_failure_count < 5:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                face_rect = (x, y, w, h)
                self.tracking_failure_count = 0
                updated = True
            else:
                self.tracking_failure_count += 1

        # Detection logic
        if not updated and (self.frame_counter % self.detection_interval == 0):
            faces = self.detect_faces(frame)
            if faces:
                x, y, w, h, _ = max(faces, key=lambda f: f[2]*f[3])
                face_rect = (x, y, w, h)
                self.tracker = cv2.legacy.TrackerMOSSE_create() 
                self.tracker.init(frame, (x, y, w, h))
                self.tracking_failure_count = 0
                updated = True

        # Process face if detected
        if face_rect:
            x, y, w, h = face_rect
            y1, y2 = max(0, y), min(frame.shape[0], y+h)
            x1, x2 = max(0, x), min(frame.shape[1], x+w)
            
            # Start async processing
            if self.frame_counter % 2 == 0:
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size > 0:
                    threading.Thread(target=self.process_face, args=(face_roi,)).start()
            
            # Draw UI
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label, conf = self.last_result
            cv2.putText(frame, f"{label} ({conf}%)", 
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            
        return frame

def real_time_detection():
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed = recognizer.process_frame(frame)
            cv2.imshow('Face Recognition', processed)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(True)
    real_time_detection()