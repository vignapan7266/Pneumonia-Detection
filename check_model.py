import tensorflow as tf

model = tf.keras.models.load_model("pneumonia_densenet121.keras")

model.summary()

print("\nLayers:")
for i, layer in enumerate(model.layers):
    print(i, layer.name)