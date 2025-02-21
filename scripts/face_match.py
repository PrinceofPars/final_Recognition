import cv2
import numpy as np
import tensorflow as tf
import h5py
from mtcnn import MTCNN
from collections import deque
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceRecognizer:
    def __init__(self, emb_file='embeddings.h5', match_thresh=0.8):
        # Initialize MTCNN detector
        self.face_detector = MTCNN()
        
        # Load MobileFaceNet with proper input shape
        self.face_net = tf.keras.models.load_model(
            'models/mobile_facenet.h5',
            custom_objects={'l2_normalize': lambda x: tf.math.l2_normalize(x, axis=-1)},
            compile=False
        )
        
        # Verify model input shape
        if self.face_net.input_shape[1:3] != (160, 160):
            raise ValueError("Model expects input shape %s, got (160, 160, 3)" % 
                           str(self.face_net.input_shape))
        
        # Warmup model with correct dimensions
        self.face_net.predict(np.zeros((1, 160, 160, 3)))
        
        # Load embeddings database
        with h5py.File(emb_file, 'r') as hf:
            self.embeddings = hf['embeddings'][:]
            self.names = [os.path.splitext(os.path.basename(p.decode()))[0] 
                        for p in hf['paths'][:]]
            
        # Recognition parameters
        self.match_thresh = match_thresh
        self.frame_skip = 3
        self.frame_counter = 0
        self.identity_buffer = deque(maxlen=5)
        
        # Tracking state
        self.last_face = None
        self.current_label = "Initializing..."
        self.confidence = 0

    def process_frame(self, frame):
        self.frame_counter += 1
        
        # Detect faces with MTCNN
        faces = self.face_detector.detect_faces(frame)
        
        if faces:
            main_face = max(faces, key=lambda f: f['box'][2]*f['box'][3])
            x, y, w, h = main_face['box']
            
            # Ensure valid face dimensions
            h, w = max(h, 20), max(w, 20)
            self.last_face = (x, y, w, h)

            if self.frame_counter % self.frame_skip == 0:
                try:
                    # Preprocess face with boundary checks
                    y1, y2 = max(0, y), min(frame.shape[0], y+h)
                    x1, x2 = max(0, x), min(frame.shape[1], x+w)
                    face_roi = frame[y1:y2, x1:x2]
                    
                    # Convert and resize properly
                    resized = cv2.resize(face_roi, (160, 160))
                    resized = (resized.astype('float32') - 127.5) / 128.0
                    
                    # Get embedding
                    embedding = self.face_net.predict(resized[None, ...], verbose=0)[0]
                    
                    # Calculate similarities
                    similarities = np.dot(self.embeddings, embedding)
                    max_idx = np.argmax(similarities)
                    max_sim = similarities[max_idx]
                    
                    # Update identity buffer
                    if max_sim > self.match_thresh:
                        self.identity_buffer.append(max_idx)
                    else:
                        self.identity_buffer.append(-1)
                    
                    # Consensus logic
                    if self.identity_buffer:
                        from collections import Counter
                        counts = Counter(self.identity_buffer)
                        if -1 in counts:
                            del counts[-1]
                        if counts:
                            final_idx = counts.most_common(1)[0][0]
                            self.current_label = self.names[final_idx]
                            self.confidence = int(similarities[final_idx] * 100)
                        else:
                            self.current_label = "Unknown"
                            self.confidence = 0
                except Exception as e:
                    print(f"Processing error: {str(e)}")

            # Draw UI elements
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{self.current_label} ({self.confidence}%)", 
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Quality: {main_face['confidence']:.2f}", 
                      (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            
        elif self.last_face:
            x, y, w, h = self.last_face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, "No face detected", (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            self.last_face = None

        return frame

def real_time_detection():
    recognizer = FaceRecognizer()
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            processed = recognizer.process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Face Recognition', cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(True)
    cv2.setUseOptimized(True)
    real_time_detection()
