# test_tf.py
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Test GPU
if tf.config.list_physical_devices('GPU'):
    print("🎉 GPU is working!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"GPU computation result:\n{c}")
else:
    print("❌ No GPU detected")