import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import h5py
from sklearn.preprocessing import Normalizer

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import h5py

@tf.keras.utils.register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=-1)

# Configuration
DETECTOR = MTCNN()
FACE_NET = tf.keras.models.load_model(
    'models/mobile_facenet.h5',
    custom_objects={'l2_normalize': l2_normalize},
    compile=False
)
EMBEDDING_SIZE = FACE_NET.output_shape[-1]  # 1024

L2_NORMALIZER = Normalizer('l2')

# Rest of your existing code remains the same...


def get_largest_face(faces):
    """Select face with largest bounding box area"""
    if not faces:
        return None
    areas = [f['box'][2] * f['box'][3] for f in faces]
    return faces[np.argmax(areas)]

def process_image(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    faces = DETECTOR.detect_faces(img)
    
    if not faces:
        return None
    
    face_data = get_largest_face(faces)
    if not face_data:
        return None
    
    x, y, w, h = face_data['box']
    face = img[y:y+h, x:x+w]
    
    # Preprocess for FaceNet
    resized = cv2.resize(face, (160, 160)).astype('float32')
    resized = (resized - 127.5) / 128.0
    
    # Generate embedding
    embedding = FACE_NET.predict(np.expand_dims(resized, 0), verbose=0)[0]
    
    return {
        'embedding': embedding,
        'centroid': (x + w//2, y + h//2),
        'path': image_path
    }

def create_embeddings_dataset(image_paths):
    with h5py.File('embeddings.h5', 'w') as hf:
        embeddings_ds = hf.create_dataset(
            'embeddings', 
            shape=(0, 1024),
            maxshape=(None, 1024),
            dtype=np.float32,
            compression='gzip'
        )
        centroids_ds = hf.create_dataset(
            'centroids',
            shape=(0, 2),
            maxshape=(None, 2),
            dtype=np.int32,
            compression='gzip'
        )
        paths_ds = hf.create_dataset(
            'paths',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype()
        )
        
        valid_count = 0
        for path in image_paths:
            result = process_image(path)
            if result is None:  # Explicit None check
                print(f"No face found in {path}")
                continue
            
            # Assign values from dictionary
            embeddings_ds.resize((valid_count+1, 1024))
            centroids_ds.resize((valid_count+1, 2))
            paths_ds.resize((valid_count+1,))
            
            embeddings_ds[valid_count] = result['embedding']
            centroids_ds[valid_count] = result['centroid']
            paths_ds[valid_count] = result['path']
            valid_count += 1

        print(f"Created embeddings for {valid_count}/{len(image_paths)} images")

if __name__ == "__main__":
    import glob
    image_files = glob.glob('face_images/*.jpg') + glob.glob('face_images/*.png')
    create_embeddings_dataset(image_files)
