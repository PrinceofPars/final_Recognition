# import tensorflow as tf
# from tensorflow.keras.applications import MobileNet, MobileNetV2
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.models import Model

# def l2_normalize(x):
#     return tf.math.l2_normalize(x, axis=1)

# # Face Recognition Model (FaceNet)
# def create_face_net():
#     base = MobileNet(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(160, 160, 3),
#         pooling='avg'
#     )
#     embeddings = Lambda(l2_normalize, name='l2_norm')(base.output)
#     return Model(inputs=base.input, outputs=embeddings)

# # Spoof Detection Model
# def create_spoof_model():
#     base = MobileNetV2(
#         weights='imagenet',
#         include_top=False,
#         input_shape=(224, 224, 3),
#         pooling='avg'
#     )
#     x = tf.keras.layers.Dense(1, activation='sigmoid')(base.output)
#     return Model(inputs=base.input, outputs=x)

# # Model Conversion Function
# def convert_models():
#     face_net = create_face_net()
#     spoof_model = create_spoof_model()
    
#     # Save Keras models
#     face_net.save('mobile_facenet.h5')
#     spoof_model.save('spoof_detector.h5')
    
#     # Convert to TFLite
#     converter = tf.lite.TFLiteConverter.from_keras_model(face_net)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_face_net = converter.convert()
#     with open('mobile_facenet.tflite', 'wb') as f:
#         f.write(tflite_face_net)
    
#     converter = tf.lite.TFLiteConverter.from_keras_model(spoof_model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     tflite_spoof = converter.convert()
#     with open('spoof_detector.tflite', 'wb') as f:
#         f.write(tflite_spoof)

# if __name__ == "__main__":
#     convert_models()
#     print("Models generated successfully!")
#     print("Files created:", 
#           "mobile_facenet.h5", 
#           "spoof_detector.h5",
#           "mobile_facenet.tflite",
#           "spoof_detector.tflite")



import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

@tf.keras.utils.register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=-1)

def create_face_net():
    base = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3),
        pooling='avg'
    )
    embeddings = Lambda(l2_normalize, name='l2_normalize')(base.output)
    return Model(inputs=base.input, outputs=embeddings)

# Save the model after regenerating with this code
