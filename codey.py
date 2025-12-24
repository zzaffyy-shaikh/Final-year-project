import os
import cv2
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from mobile_sam import SamPredictor, sam_model_registry

# ================== CONFIGURATION ==================
# Paths
IMAGE_PATH = r"C:\Users\RTC\Downloads\fyp\dataste\WhatsApp Image 2025-12-23 at 9.26.10 PM (1).jpeg"
CONVNEXT_PATH = r"C:\Users\RTC\Downloads\fyp\inference\best_convnext_model.h5"
SAM_CHECKPOINT = r"C:\Users\RTC\Downloads\fyp\mobile sam\mobile_sam.pt"
OUTPUT_PATH = r"C:\Users\RTC\Downloads\fyp\pipeline\final_precise_highlight.png"

# Labels
CLASS_NAMES = ['Atopic Dermatitis', 'Eczema', 'Melanoma', 'Psoriasis', 'Ringworm']
# ===================================================

# --- Custom ConvNeXt Layer ---
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, init_values=1e-6, projection_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
    def build(self, input_shape):
        p_dim = self.projection_dim if self.projection_dim else input_shape[-1]
        self.gamma = self.add_weight(shape=(p_dim,), 
                                    initializer=tf.keras.initializers.Constant(self.init_values),
                                    trainable=True)
    def call(self, x):
        return x * self.gamma
    def get_config(self):
        config = super().get_config()
        config.update({"init_values": self.init_values, "projection_dim": self.projection_dim})
        return config

def run_pipeline():
    print(f"--- Pipeline Started ---")

    # ---------------------------------------------------------
    # PART 1: MOBILE SAM (Segmentation & Highlighting)
    # ---------------------------------------------------------
    print("Stage 1: Running Mobile SAM Segmentation...")
    sam = sam_model_registry["vit_t"](checkpoint=SAM_CHECKPOINT)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)

    img_cv2 = cv2.imread(IMAGE_PATH)
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_sam_resized = cv2.resize(img_rgb, (1024, 1024))
    
    predictor.set_image(img_sam_resized)
    
    # Define bounding box for lesion detection
    h, w = 1024, 1024
    box_margin = 300
    input_box = np.array([box_margin, box_margin, w - box_margin, h - box_margin])

    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
    mask = masks[0].astype(np.uint8)

    # Post-process mask for smooth black outline
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask * 255, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

    highlighted = img_sam_resized.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(highlighted, contours, -1, (0, 0, 0), 6) # Bold black line
    
    Image.fromarray(highlighted).save(OUTPUT_PATH)
    print(f"✔ Segmented image saved to: {OUTPUT_PATH}")

    # ---------------------------------------------------------
    # PART 2: CONVNEXT (Classification)
    # ---------------------------------------------------------
    print("\nStage 2: Running ConvNeXt Diagnosis...")
    custom_objects = {"LayerScale": LayerScale}
    
    try:
        model = tf.keras.models.load_model(CONVNEXT_PATH, custom_objects=custom_objects, compile=False)
        
        # Classification usually uses 512x512
        img_keras = image.load_img(IMAGE_PATH, target_size=(512, 512))
        img_array = image.img_to_array(img_keras)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array, verbose=0)
        score = tf.nn.softmax(preds[0]).numpy()
        top_2_indices = np.argsort(score)[::-1][:2]

        print("\n" + "="*40)
        print("         DIAGNOSIS RESULTS")
        print("="*40)
        for i, idx in enumerate(top_2_indices):
            label = CLASS_NAMES[idx]
            conf = score[idx] * 100
            print(f"{i+1}. {label:<20} | {conf:>6.2f}%")
        print("="*40)

        # ---------------------------------------------------------
        # PART 3: VISUALIZATION
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 5))
        
        # Show Original with Highlight
        plt.subplot(1, 2, 1)
        plt.imshow(highlighted)
        plt.title("Segmented Lesion")
        plt.axis('off')

        # Show Prediction Text
        plt.subplot(1, 2, 2)
        result_text = f"Primary: {CLASS_NAMES[top_2_indices[0]]}\n({score[top_2_indices[0]]*100:.1f}%)"
        plt.text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center', fontweight='bold')
        plt.axis('off')
        plt.title("Diagnosis")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Classification Error: {e}")

if __name__ == "__main__":
    run_pipeline()