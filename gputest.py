import tensorflow as tf

print("Dispositivi disponibili:", tf.config.list_physical_devices())
print("GPU rilevate:", tf.config.list_physical_devices('GPU'))
print("TensorFlow usa GPU?", tf.test.is_built_with_cuda())
print("GPU disponibile per TensorFlow?", tf.test.is_gpu_available())
