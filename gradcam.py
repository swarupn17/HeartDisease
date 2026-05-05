import tensorflow as tf
import numpy as np
import cv2


def get_last_conv_layer_name(model):
    """Return the last Conv2D layer name in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model.")

def get_gradcam(model, img_array, last_conv_layer_name):
    """
    Generate Grad-CAM heatmap dynamically aligned with original input dimensions.
    
    Args:
        model: Keras model
        img_array: Input spectrogram batch of shape (1, height, width, channels)
        last_conv_layer_name: Name of the last convolutional layer to visualize
    
    Returns:
        heatmap_resized: Normalized heatmap of exact shape (height, width)
    """
    
    # 1. Dynamically extract the EXACT original dimensions
    # Assuming standard NHWC format: (batch, height, width, channels)
    _, target_height, target_width, _ = img_array.shape
    print(f"🔍 DEBUG - Target Audio Spectrogram Dimensions: {target_width}x{target_height} (WxH)")

    # 2. Map input to last conv layer and model output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 3. Compute gradients of the predicted class with respect to conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Handle potential list outputs
        if isinstance(predictions, list):
            predictions = predictions[0]
            
        # Get the prediction loss for the top predicted class
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # 4. Extract gradients and average them spatially
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Generate weighted feature map
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    print(f"🔍 DEBUG - Raw Conv Feature Map Shape: {heatmap.shape}")

    # 7. 🔥 CRITICAL FIX: Resize to exact original spectrogram dimensions
    # cv2.resize expects the target size as (width, height)
    heatmap_resized = cv2.resize(heatmap, (target_width, target_height))
    print(f"🔍 DEBUG - Final Resized Heatmap Shape: {heatmap_resized.shape}")

    return heatmap_resized


def save_superimposed_gradcam(img_array, heatmap, output_path="gradcam_output.png", alpha=0.4):
    """
    Applies a colormap to the heatmap and overlays it directly onto the original image.
    This ensures no clipping and a perfect visual representation.
    """
    # Grab the original image (remove batch dimension)
    original_img = img_array[0]
    
    # Scale image to 0-255 if it is normalized
    if np.max(original_img) <= 1.0:
        original_img = np.uint8(255 * original_img)
    else:
        original_img = np.uint8(original_img)
        
    # If original image is grayscale (1 channel), convert to RGB so heatmap blends well
    if original_img.shape[-1] == 1:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use OpenCV to apply the Jet colormap
    jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the original image
    superimposed_img = cv2.addWeighted(jet_heatmap, alpha, original_img, 1 - alpha, 0)

    # Save the final image to send to the React frontend
    cv2.imwrite(output_path, superimposed_img)
    print(f"✅ SUCCESS - Grad-CAM saved successfully to {output_path}")
    
    return output_path