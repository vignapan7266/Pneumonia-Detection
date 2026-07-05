import tensorflow as tf
import numpy as np
import cv2


def get_last_conv_layer(model):
    """
    Automatically finds the last Conv2D layer
    inside the DenseNet121 base model.
    """
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(img_array, model):

    base_model = model.layers[0]

    last_conv_layer_name = get_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            base_model.get_layer(last_conv_layer_name).output,
            base_model.output
        ]
    )

    classifier_input = tf.keras.Input(
        shape=grad_model.output[0].shape[1:]
    )

    x = classifier_input

    for layer in model.layers[1:]:
        x = layer(x)

    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        tape.watch(conv_outputs)

        preds = classifier_model(conv_outputs)

        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def save_gradcam(image_path, model, output_path):

    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    heatmap = make_gradcam_heatmap(img_array, model)

    original = cv2.imread(image_path)

    original = cv2.resize(original, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    cv2.imwrite(output_path, superimposed)

    return output_path