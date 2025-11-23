# main.py

"""
Secure Plant Disease Detection Backend (Single File)
---------------------------------------------------

Features:
- User auth with JWT + CSRF
- Device registration with public key (Expo app)
- Image integrity via RSA signature (client signs, backend verifies)
- Full plant disease detection pipeline (leaf/fruit):
    * is_plant_part()
    * classify_plant_type()
    * predict_leaf_disease() / predict_fruit_disease()
    * recommendations
- Single endpoint /predict used by your React Native app
"""

import base64
import io
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Request,
    status,
)
from fastapi.responses import PlainTextResponse
from jose import JWTError, jwt
from PIL import Image

import warnings

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

# Add this import at the top of your main.py file (with other imports)

warnings.filterwarnings("ignore")
from deep_translator import GoogleTranslator
# =========================
# CONFIG
# =========================
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "zh-CN": "Chinese Simplified (简体中文)",
    "zh-TW": "Chinese Traditional (繁體中文)",
    "ar": "Arabic (العربية)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "mr": "Marathi (मराठी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "gu": "Gujarati (ગુજરાતી)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "ur": "Urdu (اردو)",
    "vi": "Vietnamese (Tiếng Việt)",
    "th": "Thai (ไทย)",
    "id": "Indonesian (Bahasa Indonesia)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
    "pl": "Polish (Polski)",
    "tr": "Turkish (Türkçe)",
    "sv": "Swedish (Svenska)",
    "fi": "Finnish (Suomi)",
}

def translate_text(text: str, target_lang: str) -> str:
    """
    Translate text to target language using Google Translate
    Falls back to English if translation fails
    """
    if target_lang == "en" or not target_lang:
        return text
    
    try:
        translator = GoogleTranslator(source='en', target=target_lang)
        return translator.translate(text)
    except Exception as e:
        print(f"⚠ Translation error for {target_lang}: {e}")
        return text  # Return original English text if translation fails

def translate_list(items: list, target_lang: str) -> list:
    """Translate a list of strings"""
    if target_lang == "en" or not target_lang:
        return items
    
    translated = []
    for item in items:
        translated.append(translate_text(item, target_lang))
    return translated
# ============================================================
# 0️⃣ BASIC SECURITY CONFIG (JWT + CSRF)
# ============================================================

SECRET_KEY = "YOUR_SUPER_SECRET_JWT_KEY_123"  # change in prod
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

CSRF_SECRET = "YOUR_CSRF_SECRET_456"  # not strictly needed, but kept

# In-memory "DB" for MVP
users_db = {}        # username -> {password}
devices_db = {}      # device_id -> public_key_pem
csrf_tokens = {}     # username -> csrf_token


def create_access_token(data: dict, expires_delta: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_jwt(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid JWT")


def generate_csrf_token(username: str):
    import secrets

    token = secrets.token_hex(32)
    csrf_tokens[username] = token
    return token


def verify_csrf(username: str, token: str):
    if username not in csrf_tokens:
        raise HTTPException(status_code=401, detail="CSRF token missing")
    if csrf_tokens[username] != token:
        raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return True


# ============================================================
# 1️⃣ ML MODELS & HELPERS (YOUR EXISTING LOGIC)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        # NOTE: fixed __init__ name so PyTorch works
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            ),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4),
        )

        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            ),
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = out + self.res1(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out + self.res2(out)
        out = self.classifier(out)
        return out


# ------------------------------
# LOAD MODELS
# ------------------------------

# You will adjust these paths as needed
LEAF_MODEL_PATH = "plant-disease-model.pth"
FRUIT_MODEL_PATH = "ResNet50_final.keras"

leaf_class_names = [
    "Apple__Apple_scab",
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple__healthy",
    "Blueberry__healthy",
    "Cherry(including_sour)Powdery_mildew",
    "Cherry(including_sour)_healthy",
    "Corn_(maize)Cercospora_leaf_spot Gray_leaf_spot",
    "Corn(maize)Common_rust",
    "Corn_(maize)Northern_Leaf_Blight",
    "Corn(maize)healthy",
    "Grape__Black_rot",
    "Grape__Esca(Black_Measles)",
    "Grape__Leaf_blight(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange__Haunglongbing(Citrus_greening)",
    "Peach__Bacterial_spot",
    "Peach__healthy",
    "Pepper,bell_Bacterial_spot",
    "Pepper,_bell_healthy",
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato_healthy",
    "Raspberry_healthy",
    "Soybean__healthy",
    "Squash__Powdery_mildew",
    "Strawberry_Leaf_scorch",
    "Strawberry__healthy",
    "Tomato__Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato__Leaf_Mold",
    "Tomato__Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato__healthy",
]

leaf_model = ResNet9(3, len(leaf_class_names)).to(device)
leaf_model.load_state_dict(torch.load(LEAF_MODEL_PATH, map_location=device))
leaf_model.eval()
print("✓ Leaf model loaded successfully (PyTorch)")

fruit_class_names = [
    "APPLE_Blotch_Apple",
    "APPLE_Healthy_Apple",
    "APPLE_Rot_Apple",
    "APPLE_Scab_Apple",
    "GUAVA_Anthracnose_Guava",
    "GUAVA_Fruitfly_Guava",
    "GUAVA_Healthy_Guava",
    "MANGO_Alternaria_Mango",
    "MANGO_Anthracnose_Mango",
    "MANGO_Healthy_Mango",
    "MANGO_Stem and Rot (Lasiodiplodia)_Mango",
    "POMEGRANATE_Alternaria_Pomegranate",
    "POMEGRANATE_Anthracnose_Pomegranate",
    "POMEGRANATE_Bacterial_Blight_Pomegranate",
    "POMEGRANATE_Cercospora_Pomegranate",
    "POMEGRANATE_Healthy_Pomegranate",
]

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

try:
    fruit_model = load_model(FRUIT_MODEL_PATH)
    print("✓ Fruit model loaded successfully (full model)")
except Exception as e:
    print(f"⚠ Could not load fruit model: {e}")

# ------------------------------
# DISEASE RECOMMENDATIONS DB
# (unchanged from your code)
# ------------------------------
import cv2
from typing import Tuple

# ============================================================
# GRAD-CAM IMPLEMENTATION FOR EXPLAINABLE AI
# ============================================================




def find_last_conv_layer(model):
    """
    Robustly find the last convolutional layer in a model
    """
    def is_conv_layer(layer):
        try:
            if hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                if shape is not None and len(shape) == 4:
                    return True
            
            layer_type = type(layer).__name__
            if 'Conv' in layer_type:
                return True
            
            if hasattr(layer, 'get_config'):
                config = layer.get_config()
                if 'filters' in config:
                    return True
            
            return False
        except Exception as e:
            print(f"   ⚠ Error checking layer {layer.name}: {e}")
            return False
    
    # First, try to find a base model (common in transfer learning)
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            print(f"📦 Found nested model: {layer.name}")
            for nested_layer in reversed(layer.layers):
                if is_conv_layer(nested_layer):
                    print(f"   ✓ Found conv layer in nested model: {nested_layer.name}")
                    print(f"      Type: {type(nested_layer).__name__}")
                    return nested_layer, layer
    
    # If no nested model, search in main model
    for layer in reversed(model.layers):
        if is_conv_layer(layer):
            print(f"   ✓ Found conv layer in main model: {layer.name}")
            print(f"      Type: {type(layer).__name__}")
            return layer, model
    
    return None, None





def predict_fruit_disease(image_path, generate_xai=True):
    """Predict disease for fruit images with optional Grad-CAM"""
    print(f"\n🍎 Starting FRUIT disease prediction...")
    print(f"   - Image path: {image_path}")
    print(f"   - Generate XAI: {generate_xai}")
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array_original = np.array(img)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_processed = resnet50_preprocess(img_array.copy())
    
    # Convert to TensorFlow tensor
    img_array_tensor = tf.convert_to_tensor(img_array_processed, dtype=tf.float32)

    # Prediction
    pred = fruit_model.predict(img_array_processed, verbose=0)[0]
    label_idx = np.argmax(pred)
    confidence = float(pred[label_idx])
    predicted_class = fruit_class_names[label_idx]
    
    print(f"   - Predicted: {predicted_class}")
    print(f"   - Confidence: {confidence:.4f}")
    
    # Generate Grad-CAM if requested
    gradcam_overlay = None
    if generate_xai:
        try:
            print("🎨 Generating Grad-CAM for FRUIT model...")
            
            # Try simple approach first
            heatmap = generate_gradcam_keras_simple(
                fruit_model,
                img_array_tensor,
                label_idx
            )
            
            # If that fails, try the complex approach
            if heatmap is None:
                print("   🔄 Trying alternative method...")
                heatmap = generate_gradcam_keras_simple(
                    fruit_model,
                    img_array_tensor,
                    label_idx
                )
            
            if heatmap is not None:
                print(f"   - Heatmap shape: {heatmap.shape}")
                print(f"   - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                
                gradcam_overlay = overlay_heatmap_on_image(img_array_original, heatmap)
                gradcam_overlay_b64 = image_to_base64(gradcam_overlay)
                
                print(f"✅ Grad-CAM generated successfully!")
                print(f"   - Base64 length: {len(gradcam_overlay_b64)} chars")
                
                return predicted_class, confidence, gradcam_overlay_b64
            else:
                print("⚠️ Both Grad-CAM methods failed")
                
        except Exception as e:
            print(f"❌ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam_overlay = None
    
    return predicted_class, confidence, gradcam_overlay

import tensorflow as tf









disease_recommendations = {
    # Leaf Diseases
    "Apple_scab": [
        "🔹 Apply fungicides containing captan or sulfur during early spring",
        "🔹 Remove and destroy fallen leaves to reduce overwintering spores",
        "🔹 Plant resistant apple varieties like Liberty or Freedom",
    ],
    "Black_rot": [
        "🔹 Prune infected branches and remove mummified fruits",
        "🔹 Apply copper-based fungicides during dormant season",
        "🔹 Maintain good air circulation by proper pruning",
    ],
    "Cedar_apple_rust": [
        "🔹 Remove nearby cedar or juniper trees if possible",
        "🔹 Apply fungicides containing myclobutanil in early spring",
        "🔹 Plant resistant apple varieties",
    ],
    "Powdery_mildew": [
        "🔹 Apply sulfur-based or potassium bicarbonate sprays",
        "🔹 Improve air circulation around plants",
        "🔹 Remove infected plant parts immediately",
    ],
    "Cercospora_leaf_spot": [
        "🔹 Apply chlorothalonil or mancozeb fungicides",
        "🔹 Practice crop rotation (avoid planting corn in same area)",
        "🔹 Remove and destroy infected crop debris",
    ],
    "Common_rust": [
        "🔹 Plant resistant corn hybrids",
        "🔹 Apply azoxystrobin or propiconazole fungicides",
        "🔹 Ensure proper plant spacing for air circulation",
    ],
    "Northern_Leaf_Blight": [
        "🔹 Use resistant corn varieties",
        "🔹 Apply fungicides at early infection stage",
        "🔹 Practice minimum 2-year crop rotation",
    ],
    "Esca_(Black_Measles)": [
        "🔹 Prune infected wood during dry weather",
        "🔹 Apply wound protectants after pruning",
        "🔹 No chemical cure available - focus on prevention",
    ],
    "Leaf_blight": [
        "🔹 Apply copper-based fungicides",
        "🔹 Remove infected leaves and destroy them",
        "🔹 Avoid overhead watering",
    ],
    "Haunglongbing_(Citrus_greening)": [
        "🔹 Remove and destroy infected trees immediately",
        "🔹 Control Asian citrus psyllid vector with insecticides",
        "🔹 Plant disease-free certified nursery stock only",
    ],
    "Bacterial_spot": [
        "🔹 Apply copper-based bactericides",
        "🔹 Use disease-free seeds and transplants",
        "🔹 Avoid working with plants when wet",
    ],
    "Early_blight": [
        "🔹 Apply chlorothalonil or mancozeb fungicides",
        "🔹 Remove lower leaves that touch the ground",
        "🔹 Practice crop rotation (3-4 years)",
    ],
    "Late_blight": [
        "🔹 Apply fungicides containing chlorothalonil or mancozeb",
        "🔹 Remove and destroy all infected plant material immediately",
        "🔹 Improve air circulation and avoid overhead irrigation",
    ],
    "Leaf_Mold": [
        "🔹 Increase ventilation in greenhouse or garden",
        "🔹 Apply chlorothalonil fungicide",
        "🔹 Avoid overhead watering",
    ],
    "Septoria_leaf_spot": [
        "🔹 Apply fungicides containing chlorothalonil",
        "🔹 Mulch around plants to prevent soil splash",
        "🔹 Remove infected lower leaves",
    ],
    "Spider_mites": [
        "🔹 Spray with insecticidal soap or neem oil",
        "🔹 Increase humidity around plants",
        "🔹 Introduce predatory mites as biological control",
    ],
    "Target_Spot": [
        "🔹 Apply mancozeb or chlorothalonil fungicides",
        "🔹 Practice crop rotation",
        "🔹 Remove plant debris after harvest",
    ],
    "Tomato_Yellow_Leaf_Curl_Virus": [
        "🔹 Control whitefly vectors with insecticides or yellow sticky traps",
        "🔹 Remove and destroy infected plants",
        "🔹 Use virus-resistant tomato varieties",
    ],
    "Tomato_mosaic_virus": [
        "🔹 Remove and destroy infected plants immediately",
        "🔹 Disinfect tools and hands to prevent spread",
        "🔹 Plant resistant varieties and use virus-free seeds",
    ],
    "Leaf_scorch": [
        "🔹 Ensure adequate watering during dry periods",
        "🔹 Apply fungicides if fungal infection is present",
        "🔹 Mulch to retain soil moisture",
    ],
    # Fruit Diseases
    "Anthracnose": [
        "🔹 Apply copper-based or mancozeb fungicides before rainy season",
        "🔹 Remove and destroy infected fruits and plant debris",
        "🔹 Improve orchard drainage and air circulation",
    ],
    "Fruitfly": [
        "🔹 Use pheromone traps to monitor and control adult flies",
        "🔹 Bag fruits with paper or cloth bags",
        "🔹 Remove and destroy infested fallen fruits immediately",
    ],
    "Alternaria": [
        "🔹 Apply fungicides containing azoxystrobin or difenoconazole",
        "🔹 Prune to improve air circulation",
        "🔹 Remove infected plant parts and destroy them",
    ],
    "Bacterial_Blight": [
        "🔹 Apply copper-based bactericides",
        "🔹 Prune infected branches during dry weather",
        "🔹 Avoid overhead irrigation",
    ],
    "Cercospora": [
        "🔹 Apply mancozeb or chlorothalonil fungicides",
        "🔹 Remove fallen leaves and fruit debris",
        "🔹 Maintain proper plant spacing",
    ],
    "Blotch": [
        "🔹 Apply captan or thiophanate-methyl fungicides",
        "🔹 Prune to reduce humidity within canopy",
        "🔹 Remove mummified fruits",
    ],
    "Rot": [
        "🔹 Improve storage conditions (cool, dry)",
        "🔹 Handle fruits carefully to avoid wounds",
        "🔹 Apply post-harvest fungicides if needed",
    ],
    "Scab": [
        "🔹 Apply fungicides during susceptible growth stages",
        "🔹 Remove leaf litter in fall",
        "🔹 Plant resistant varieties",
    ],
    "Stem_and_Rot": [
        "🔹 Prune infected branches below diseased area",
        "🔹 Apply copper fungicides to wounds",
        "🔹 Improve tree vigor through proper fertilization",
    ],
    "Lasiodiplodia": [
        "🔹 Apply thiophanate-methyl or propiconazole",
        "🔹 Avoid tree stress through proper irrigation",
        "🔹 Prune and destroy infected branches",
    ],
}

# ------------------------------
# HELPER FUNCTIONS (unchanged)
# ------------------------------


def is_plant_part(image_array):
    """
    Enhanced heuristic to check if image contains plant material
    Detects both LEAVES (green) and FRUITS (various colors)
    """
    if len(image_array.shape) == 4:
        image_array = image_array[0]

    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)

    green_channel = image_array[:, :, 1].astype(np.float32)
    red_channel = image_array[:, :, 0].astype(np.float32)
    blue_channel = image_array[:, :, 2].astype(np.float32)

    total_pixels = green_channel.size

    # 1. Check green dominance (for LEAVES)
    green_pixels = np.sum(
        (green_channel > red_channel + 10) & (green_channel > blue_channel + 10)
    )
    green_ratio = green_pixels / total_pixels

    # 2. Check for FRUIT colors (red, yellow, orange, brown)
    red_fruit_pixels = np.sum(
        (red_channel > 120)
        & (red_channel > green_channel + 20)
        & (red_channel > blue_channel + 20)
    )
    yellow_fruit_pixels = np.sum(
        (red_channel > 100)
        & (green_channel > 80)
        & (blue_channel < 120)
        & (np.abs(red_channel - green_channel) < 60)
    )
    green_fruit_pixels = np.sum(
        (green_channel > 100)
        & (green_channel > red_channel)
        & (green_channel > blue_channel)
    )
    fruit_color_ratio = (
        red_fruit_pixels + yellow_fruit_pixels + green_fruit_pixels
    ) / total_pixels

    max_rgb = np.maximum(
        np.maximum(red_channel, green_channel),
        blue_channel,
    )
    min_rgb = np.minimum(
        np.minimum(red_channel, green_channel),
        blue_channel,
    )
    saturation = (max_rgb - min_rgb) / (max_rgb + 1e-6)
    avg_saturation = np.mean(saturation)

    texture_variance = np.var(green_channel)

    skin_pixels = np.sum(
        (red_channel > 95)
        & (red_channel < 220)
        & (green_channel > 40)
        & (green_channel < 180)
        & (blue_channel > 20)
        & (blue_channel < 150)
        & (red_channel > green_channel)
        & (red_channel > blue_channel)
        & (np.abs(red_channel - green_channel) < 50)
    )
    skin_ratio = skin_pixels / total_pixels

    color_uniformity = 1.0 - avg_saturation
    is_too_uniform = (color_uniformity > 0.7) and (texture_variance < 500)

    is_leaf = (green_ratio > 0.20) and (texture_variance > 600)
    is_fruit = (
        (fruit_color_ratio > 0.30)
        and (avg_saturation > 0.15)
        and (texture_variance > 400)
    )
    is_rejected = (skin_ratio > 0.25) or is_too_uniform

    is_plant = (is_leaf or is_fruit) and not is_rejected
    return is_plant


def classify_plant_type(image_array, confidence_threshold=0.3):
    """
    Classify if image is LEAF or FRUIT (heuristic)
    """
    if len(image_array.shape) == 4:
        image_array = image_array[0]

    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)

    green_channel = image_array[:, :, 1]
    red_channel = image_array[:, :, 0]

    green_mean = np.mean(green_channel)
    red_mean = np.mean(red_channel)

    green_dominance = green_mean / (red_mean + 1e-6)

    if green_dominance > 1.3:
        return "LEAF", 0.8
    elif red_mean > green_mean * 1.2:
        return "FRUIT", 0.7
    else:
        return "LEAF", 0.5

# ============================================================
# FINAL WORKING GRAD-CAM - HANDLES SEQUENTIAL MODELS PROPERLY
# ============================================================

def generate_gradcam_keras_simple(model, img_array, pred_index):
    """
    Grad-CAM for Keras models - handles Sequential models properly
    """
    print("🔍 Simple Grad-CAM approach...")
    
    # 🔥 CRITICAL: Build the model first by calling it
    try:
        print("🔨 Building model...")
        _ = model(img_array)
        print("   ✓ Model built")
    except Exception as e:
        print(f"   ⚠ Build warning: {e}")
    
    # Find last conv layer in nested models
    last_conv_layer = None
    base_model = None
    
    for layer in model.layers:
        if hasattr(layer, 'layers'):  # This is a nested model (ResNet50)
            base_model = layer
            print(f"📦 Found base model: {layer.name}")
            for nested_layer in reversed(layer.layers):
                if 'conv' in nested_layer.name.lower():
                    last_conv_layer = nested_layer
                    break
        if last_conv_layer:
            break
    
    if not last_conv_layer:
        print("❌ No conv layer found")
        return None
    
    print(f"🎯 Using layer: {last_conv_layer.name}")
    
    try:
        # 🔥 FIX: Use the base model (ResNet50) instead of full Sequential model
        # This avoids the "has never been called" error
        
        # Create model from base_model input to conv layer output
        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )
        
        print("   ✓ Gradient model created")
        
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, base_outputs = grad_model(img_array)
            
            # Pass through remaining layers (after base model)
            x = base_outputs
            for layer in model.layers:
                if layer != base_model:
                    x = layer(x)
            
            predictions = x
            loss = predictions[:, pred_index]
        
        # Get gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            print("❌ No gradients computed")
            return None
        
        print(f"   ✓ Gradients computed: {grads.shape}")
        
        # Pool gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight channels by pooled gradients
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        print(f"   ✓ Heatmap generated: {heatmap.shape}, [{heatmap.min():.3f}, {heatmap.max():.3f}]")
        return heatmap
        
    except Exception as e:
        print(f"❌ Grad-CAM error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_gradcam_pytorch(model, img_tensor, target_layer, predicted_class_idx):
    """
    Generate Grad-CAM heatmap for PyTorch model (leaf disease detection)
    """
    model.eval()
    
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    output = model(img_tensor)
    
    model.zero_grad()
    class_loss = output[0, predicted_class_idx]
    class_loss.backward()
    
    handle_backward.remove()
    handle_forward.remove()
    
    gradients_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradients_val, axis=(1, 2))
    
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activations_val[i]
    
    cam = np.maximum(cam, 0)
    
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam


def overlay_heatmap_on_image(img_array, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image
    """
    try:
        import cv2
        
        if len(img_array.shape) == 4:
            img_array = img_array[0]
        
        h, w = img_array.shape[:2]
        
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_resized = np.clip(heatmap_resized, 0, 1)
        
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        overlayed = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlayed
        
    except Exception as e:
        print(f"❌ Error in overlay_heatmap_on_image: {e}")
        import traceback
        traceback.print_exc()
        return img_array


def image_to_base64(img_array):
    """Convert numpy image array to base64 string"""
    try:
        import base64
        from io import BytesIO
        
        if img_array.dtype != np.uint8:
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        img_pil = Image.fromarray(img_array)
        
        buffered = BytesIO()
        img_pil.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return img_base64
        
    except Exception as e:
        print(f"❌ Error in image_to_base64: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict_leaf_disease(image_path, generate_xai=True):
    """Predict disease for leaf images with optional Grad-CAM"""
    print(f"\n🍃 Starting LEAF disease prediction...")
    print(f"   - Image path: {image_path}")
    print(f"   - Generate XAI: {generate_xai}")
    
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    img = Image.open(image_path).convert("RGB")
    img_array_original = np.array(img.resize((256, 256)))
    img_tensor = transform(img)
    xb = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = leaf_model(xb)
        probs = torch.nn.functional.softmax(preds, dim=1)
        conf, idx = torch.max(probs, dim=1)

    predicted_class = leaf_class_names[idx.item()]
    confidence = conf.item()
    
    print(f"   - Predicted: {predicted_class}")
    print(f"   - Confidence: {confidence:.4f}")
    
    gradcam_overlay = None
    if generate_xai:
        try:
            print("🎨 Generating Grad-CAM for LEAF model...")
            
            heatmap = generate_gradcam_pytorch(
                leaf_model, 
                xb.requires_grad_(), 
                leaf_model.conv4, 
                idx.item()
            )
            
            print(f"   - Heatmap shape: {heatmap.shape}")
            print(f"   - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
            
            gradcam_overlay = overlay_heatmap_on_image(img_array_original, heatmap)
            gradcam_overlay_b64 = image_to_base64(gradcam_overlay)
            
            print(f"✅ Grad-CAM generated successfully!")
            print(f"   - Base64 length: {len(gradcam_overlay_b64)} chars")
            
            return predicted_class, confidence, gradcam_overlay_b64
            
        except Exception as e:
            print(f"❌ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam_overlay = None

    return predicted_class, confidence, gradcam_overlay


def predict_fruit_disease(image_path, generate_xai=True):
    """Predict disease for fruit images with optional Grad-CAM"""
    print(f"\n🍎 Starting FRUIT disease prediction...")
    print(f"   - Image path: {image_path}")
    print(f"   - Generate XAI: {generate_xai}")
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array_original = np.array(img)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array_processed = resnet50_preprocess(img_array.copy())
    
    # Convert to TensorFlow tensor
    img_array_tensor = tf.convert_to_tensor(img_array_processed, dtype=tf.float32)

    # Prediction
    pred = fruit_model.predict(img_array_processed, verbose=0)[0]
    label_idx = np.argmax(pred)
    confidence = float(pred[label_idx])
    predicted_class = fruit_class_names[label_idx]
    
    print(f"   - Predicted: {predicted_class}")
    print(f"   - Confidence: {confidence:.4f}")
    
    # Generate Grad-CAM if requested
    gradcam_overlay = None
    if generate_xai:
        try:
            print("🎨 Generating Grad-CAM for FRUIT model...")
            
            # Use the simple Grad-CAM approach
            heatmap = generate_gradcam_keras_simple(
                fruit_model,
                img_array_tensor,
                label_idx
            )
            
            if heatmap is not None:
                print(f"   - Heatmap shape: {heatmap.shape}")
                print(f"   - Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                
                gradcam_overlay = overlay_heatmap_on_image(img_array_original, heatmap)
                gradcam_overlay_b64 = image_to_base64(gradcam_overlay)
                
                print(f"✅ Grad-CAM generated successfully!")
                print(f"   - Base64 length: {len(gradcam_overlay_b64)} chars")
                
                return predicted_class, confidence, gradcam_overlay_b64
            else:
                print("⚠️ Grad-CAM generation returned None")
                
        except Exception as e:
            print(f"❌ Grad-CAM generation failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam_overlay = None
    
    return predicted_class, confidence, gradcam_overlay

def get_recommendations(disease_name: str, language: str = "en"):
    """
    Get treatment recommendations for a disease in specified language
    
    Args:
        disease_name: Name of the disease
        language: Language code (default: "en")
    """
    # Get English recommendations
    recs_en = []
    for key in disease_recommendations:
        if key.lower() in disease_name.lower():
            recs_en = disease_recommendations[key]
            break
    
    if not recs_en:
        recs_en = [
            "🔹 Consult with a local agricultural extension officer",
            "🔹 Remove and destroy infected plant parts",
            "🔹 Monitor plant regularly and maintain good hygiene",
        ]
    
    # If English requested, return as-is
    if language == "en":
        return recs_en
    
    # TODO: Add translation logic here if needed
    # For now, return English with a note
    print(f"   ℹ️ Translation to '{language}' not yet implemented, returning English")
    return recs_en

def get_severity_score(confidence: float, is_healthy: bool):
    """
    Calculate severity based on confidence score
    
    Args:
        confidence: Model confidence score (0-1)
        is_healthy: Whether the plant is healthy
        
    Returns:
        dict with severity level and description
    """
    confidence_percent = confidence * 100
    
    # If healthy, severity is always None
    if is_healthy:
        return {
            "severity": "Healthy",
            "severity_score": 0,
            "confidence_percent": round(confidence_percent, 2),
            "message": "Plant appears healthy"
        }
    
    # For diseased plants, calculate severity
    if confidence_percent >= 85:
        severity = "Severe"
        severity_score = 3
        message = "High confidence disease detection - immediate action recommended"
    elif confidence_percent >= 50:
        severity = "Moderate"
        severity_score = 2
        message = "Moderate disease detected - treatment advised"
    elif confidence_percent >= 40:
        severity = "Mild"
        severity_score = 1
        message = "Mild disease symptoms - monitor closely"
    else:
        severity = "Weak Detection"
        severity_score = 0
        message = "Low confidence - consider retaking image or consulting expert"
    
    return {
        "severity": severity,
        "severity_score": severity_score,
        "confidence_percent": round(confidence_percent, 2),
        "message": message
    }

def format_result(prediction, confidence, plant_type):
    """Original print-based formatter (kept for reference, not used in API)"""
    if plant_type == "LEAF":
        parts = prediction.split("_")
        plant = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ")

        print(f" Plant Type: LEAF")
        print(f" Plant: {plant}")

        if "healthy" in disease.lower():
            print(f" Status: HEALTHY")
            print(f" Confidence: {confidence*100:.2f}%")
            print("\n Your plant looks healthy! Continue good care practices:")
            print("  • Regular watering")
            print("  • Adequate sunlight")
            print("  • Monitor for any changes")
        else:
            print(f"  Disease Detected: {disease}")
            print(f" Confidence: {confidence*100:.2f}%")
            print("\n RECOMMENDED TREATMENTS:")
            recommendations = get_recommendations(disease)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

    else:
        parts = prediction.split("_")
        fruit = parts[0]
        disease = "_".join(parts[1:-1])

        print(f" Plant Type: FRUIT")
        print(f" Fruit: {fruit}")

        if "healthy" in disease.lower():
            print(f" Status: HEALTHY")
            print(f" Confidence: {confidence*100:.2f}%")
            print("\n Your fruit looks healthy! Continue good care practices:")
            print("  • Regular inspection")
            print("  • Proper storage conditions")
            print("  • Timely harvesting")
        else:
            print(f"  Disease Detected: {disease}")
            print(f" Confidence: {confidence*100:.2f}%")
            print("\n RECOMMENDED TREATMENTS:")
            recommendations = get_recommendations(disease)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")

    print("=" * 60)


# ============================================================
# 2️⃣ MAIN BACKEND PIPELINE (image_bytes -> JSON result)
# ============================================================


def run_detection_pipeline_bytes(image_bytes: bytes, generate_xai: bool = True, language: str = "en"):
    """
    Backend version of your full detection system.
    Accepts raw image bytes, returns structured JSON with optional Grad-CAM.
    
    Args:
        image_bytes: Raw image bytes
        generate_xai: Whether to generate Grad-CAM visualization
        language: Language code for recommendations
    """
    print(f"\n🔬 run_detection_pipeline_bytes() called")
    print(f"   - Image size: {len(image_bytes)} bytes")
    print(f"   - Generate XAI: {generate_xai}")
    print(f"   - Language: {language}")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # Step 1: Check plant part
    print(f"   🔍 Step 1: Checking if image contains plant...")
    is_plant = is_plant_part(img_array)
    print(f"      Result: {'Plant detected' if is_plant else 'Not a plant'}")

    if not is_plant:
        return {
            "is_plant": False,
            "message": "NOT A PLANT PART DETECTED",
            "language": language
        }

    # Step 2: classify plant type
    print(f"   🌱 Step 2: Classifying plant type...")
    plant_type, type_confidence = classify_plant_type(img_array)
    print(f"      Result: {plant_type} (confidence: {type_confidence*100:.2f}%)")

    # Step 3: Save image as a temp file
    temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
    print(f"   💾 Step 3: Saving temp file: {temp_filename}")
    
    with open(temp_filename, "wb") as f:
        f.write(image_bytes)

    try:
        # Step 4: Disease prediction
        print(f"   🔬 Step 4: Running disease prediction...")
        
        if plant_type == "LEAF":
            print(f"      Using LEAF model...")
            prediction, confidence, gradcam_overlay = predict_leaf_disease(
                temp_filename, generate_xai=generate_xai
            )
        else:
            print(f"      Using FRUIT model...")
            prediction, confidence, gradcam_overlay = predict_fruit_disease(
                temp_filename, generate_xai=generate_xai
            )
        
        print(f"      Prediction: {prediction}")
        print(f"      Confidence: {confidence*100:.2f}%")
        print(f"      Grad-CAM: {'Available' if gradcam_overlay else 'Not available'}")
        
    finally:
        # Always delete the temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            print(f"   🗑️ Temp file deleted: {temp_filename}")

    # Step 5: Get recommendations
    print(f"   📋 Step 5: Getting recommendations...")
    recs = get_recommendations(prediction, language=language)  # Pass language here
    print(f"      Recommendations: {len(recs)} items")
    
    # Check if plant is healthy
    is_healthy = "healthy" in prediction.lower()
    print(f"      Health status: {'Healthy' if is_healthy else 'Disease detected'}")
    
    # Get severity scoring
    print(f"   ⚖️ Step 6: Calculating severity...")
    severity_info = get_severity_score(confidence, is_healthy)
    print(f"      Severity: {severity_info['severity']} (score: {severity_info['severity_score']}/3)")

    # Build complete result
    result = {
        "is_plant": True,
        "plant_type": plant_type,
        "plant_type_confidence": float(type_confidence),
        "disease": prediction,
        "disease_confidence": float(confidence),
        "severity": severity_info["severity"],
        "severity_score": severity_info["severity_score"],
        "severity_message": severity_info["message"],
        "confidence_percent": severity_info["confidence_percent"],
        "recommendations": recs,
        "language": language
    }
    
    # Add Grad-CAM if available
    if gradcam_overlay:
        result["gradcam_overlay"] = gradcam_overlay
        result["xai_available"] = True
    else:
        result["gradcam_overlay"] = None
        result["xai_available"] = False
    
    print(f"   ✅ Pipeline completed successfully")
    print(f"      Result keys: {list(result.keys())}")
    
    return result

    


# ============================================================
# 3️⃣ FASTAPI APP & ROUTES (JWT + CSRF + SIGNATURE + PIPELINE)
# ============================================================

app = FastAPI(title="Secure Plant Disease Detection", version="1.0.0")


@app.get("/health", response_class=PlainTextResponse)
async def health_check():
    return "OK"


# ---------- AUTH ROUTES ----------


@app.post("/auth/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")

    # NOTE: plain text password for MVP ONLY
    users_db[username] = {"password": password}
    return {"status": "registered"}


@app.post("/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username not in users_db or users_db[username]["password"] != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": username})
    csrf_token = generate_csrf_token(username)

    return {
        "access_token": access_token,
        "csrf_token": csrf_token,
        "token_type": "bearer",
    }


# ---------- DEVICE REGISTER ----------


@app.post("/register-device")
async def register_device(
    device_id: str = Form(...),
    public_key: str = Form(...),
):
    devices_db[device_id] = public_key
    return {"status": "device_registered"}


# ---------- SIGNATURE VERIFY HELPER ----------


def verify_image_signature(image_bytes: bytes, signature_bytes: bytes, public_key_pem: str) -> bool:
    public_key = serialization.load_pem_public_key(public_key_pem.encode())

    digest = hashes.Hash(hashes.SHA256())
    digest.update(image_bytes)
    img_hash = digest.finalize()

    try:
        public_key.verify(
            signature_bytes,
            img_hash,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return True
    except Exception:
        return False


# ---------- MAIN SECURE PREDICT ROUTE ----------

@app.get("/auth/get")
async def get_users():
    return users_db

@app.get("/auth/devices")
async def get_devices():
    return devices_db

# @app.post("/predict")
# async def predict(
#     request: Request,
#     image: UploadFile = File(...),
#     signature: str = Form(...),
#     device_id: str = Form(...),
#     token: str = Form(...),
#     csrf: str = Form(...),
#     generate_xai: bool = Form(True),
# ):
#     # 1) JWT
#     payload = verify_jwt(token)
#     username = payload["sub"]

#     # 2) CSRF
#     verify_csrf(username, csrf)

#     # 3) Device & public key
#     if device_id not in devices_db:
#         raise HTTPException(status_code=400, detail="Unknown device")
#     public_key_pem = devices_db[device_id]

#     # 4) Read image
#     img_bytes = await image.read()
#     if not img_bytes:
#         raise HTTPException(status_code=400, detail="Empty image")

#     # 5) Verify signature (base64 -> bytes)
#     try:
#         raw_sig = base64.b64decode(signature)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid signature encoding")

#     if not verify_image_signature(img_bytes, raw_sig, public_key_pem):
#         raise HTTPException(status_code=400, detail="Tampered or invalid image!")

#     # 6) Run full ML pipeline
#     result = run_detection_pipeline_bytes(img_bytes, generate_xai=generate_xai)

#     return {
#         "status": "ok",
#         "integrity": "passed",
#         "result": result,
#     }
    
    
@app.get("/languages")
async def get_supported_languages():
    """Return list of supported languages for translations"""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "default": "en"
    }


@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    signature: str = Form(...),
    device_id: str = Form(...),
    token: str = Form(...),
    csrf: str = Form(...),
    generate_xai: bool = Form(True),
    language: Optional[str] = Form("en"),
):
    """
    Secure plant disease prediction endpoint with full debugging
    
    Args:
        image: Image file to analyze
        signature: RSA signature of the image
        device_id: Unique device identifier
        token: JWT authentication token
        csrf: CSRF token for request validation
        generate_xai: Whether to generate Grad-CAM visualization
        language: Language code for recommendations (default: "en")
    """
    
    # ================= DEBUG: Request Info =================
    print("\n" + "="*70)
    print("🔍 DEBUG: /predict ENDPOINT CALLED")
    print("="*70)
    print(f"📅 Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"🌐 Client IP: {request.client.host if request.client else 'Unknown'}")
    print(f"📱 User Agent: {request.headers.get('user-agent', 'Unknown')}")
    print("="*70)
    
    # ================= Validate Language =================
    print(f"\n🌍 Language Validation:")
    print(f"   - Requested language: {language}")
    print(f"   - Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    
    if language not in SUPPORTED_LANGUAGES:
        print(f"   ❌ INVALID LANGUAGE: '{language}'")
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    print(f"   ✅ Language validated: {language}")
    
    # ================= DEBUG: Input Parameters =================
    print(f"\n📥 Input Parameters:")
    print(f"   - Token (first 30): {token[:30]}...")
    print(f"   - Token length: {len(token)} chars")
    print(f"   - CSRF (first 30): {csrf[:30]}...")
    print(f"   - CSRF length: {len(csrf)} chars")
    print(f"   - Device ID: {device_id}")
    print(f"   - Signature (first 30): {signature[:30]}...")
    print(f"   - Signature length: {len(signature)} chars")
    print(f"   - Generate XAI: {generate_xai} (type: {type(generate_xai).__name__})")
    print(f"   - Language: {language}")
    print(f"   - Image filename: {image.filename}")
    print(f"   - Image content type: {image.content_type}")

    # ================= STEP 1: JWT Verification =================
    print(f"\n🔐 STEP 1: JWT Verification")
    try:
        payload = verify_jwt(token)
        username = payload["sub"]
        print(f"   ✅ JWT verified successfully")
        print(f"   - Username: {username}")
        print(f"   - Token expiry: {datetime.fromtimestamp(payload.get('exp', 0)).isoformat()}")
    except HTTPException as e:
        print(f"   ❌ JWT verification FAILED: {e.detail}")
        raise
    except Exception as e:
        print(f"   ❌ JWT verification ERROR: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid JWT token")

    # ================= STEP 2: CSRF Verification =================
    print(f"\n🛡️ STEP 2: CSRF Verification")
    try:
        verify_csrf(username, csrf)
        print(f"   ✅ CSRF token verified for user: {username}")
    except HTTPException as e:
        print(f"   ❌ CSRF verification FAILED: {e.detail}")
        raise
    except Exception as e:
        print(f"   ❌ CSRF verification ERROR: {str(e)}")
        raise HTTPException(status_code=403, detail="Invalid CSRF token")

    # ================= STEP 3: Device Registry Check =================
    print(f"\n🔑 STEP 3: Device Registry Check")
    print(f"   - Checking device_id: {device_id}")
    print(f"   - Total registered devices: {len(devices_db)}")
    print(f"   - Registered device IDs: {list(devices_db.keys())}")
    
    if device_id not in devices_db:
        print(f"   ❌ DEVICE NOT FOUND: {device_id}")
        raise HTTPException(status_code=400, detail="Unknown device. Please register device first.")
    
    public_key_pem = devices_db[device_id]
    print(f"   ✅ Device FOUND in registry")
    print(f"   - Public key length: {len(public_key_pem)} chars")
    print(f"   - Public key prefix: {public_key_pem[:50]}...")

    # ================= STEP 4: Image Reading =================
    print(f"\n📸 STEP 4: Image Reading")
    try:
        img_bytes = await image.read()
        print(f"   - Bytes read: {len(img_bytes):,} bytes ({len(img_bytes)/1024:.2f} KB)")
        print(f"   - Image size: {len(img_bytes)/1024/1024:.2f} MB")
        
        if not img_bytes:
            print(f"   ❌ EMPTY IMAGE: No bytes received")
            raise HTTPException(status_code=400, detail="Empty image file")
        
        if len(img_bytes) < 100:
            print(f"   ⚠️ WARNING: Image very small ({len(img_bytes)} bytes)")
        
        if len(img_bytes) > 10 * 1024 * 1024:  # 10MB
            print(f"   ⚠️ WARNING: Large image ({len(img_bytes)/1024/1024:.2f} MB)")
        
        print(f"   ✅ Image read successfully")
        
    except Exception as e:
        print(f"   ❌ IMAGE READ ERROR: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to read image: {str(e)}")

    # ================= STEP 5: Signature Verification =================
    print(f"\n🔏 STEP 5: Signature Verification")
    print(f"   - Signature string length: {len(signature)} chars")
    
    try:
        raw_sig = base64.b64decode(signature)
        print(f"   - Decoded signature: {len(raw_sig)} bytes")
        print(f"   - Signature bytes (first 20): {raw_sig[:20].hex()}")
    except Exception as e:
        print(f"   ❌ SIGNATURE DECODING FAILED: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid signature encoding (not valid base64)")

    print(f"   - Computing image hash...")
    try:
        # Compute hash for debugging
        from cryptography.hazmat.primitives import hashes
        digest = hashes.Hash(hashes.SHA256())
        digest.update(img_bytes)
        img_hash = digest.finalize()
        print(f"   - Image SHA256 hash: {img_hash.hex()[:40]}...")
    except Exception as e:
        print(f"   ⚠️ Hash computation for debug failed: {e}")
    
    print(f"   - Verifying signature with public key...")
    verification_start = datetime.utcnow()
    
    try:
        is_valid = verify_image_signature(img_bytes, raw_sig, public_key_pem)
        verification_time = (datetime.utcnow() - verification_start).total_seconds()
        
        if not is_valid:
            print(f"   ❌ SIGNATURE VERIFICATION FAILED")
            print(f"   - Verification time: {verification_time:.3f}s")
            print(f"   - Image may be tampered or signature is invalid")
            raise HTTPException(status_code=400, detail="Image integrity check failed - signature invalid")
        
        print(f"   ✅ Signature verification PASSED")
        print(f"   - Verification time: {verification_time:.3f}s")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"   ❌ SIGNATURE VERIFICATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Signature verification error: {str(e)}")

    # ================= STEP 6: ML Pipeline =================
    print(f"\n🌿 STEP 6: ML Disease Detection Pipeline")
    print(f"   - Generate XAI: {generate_xai}")
    print(f"   - Language: {language}")
    print(f"   - Starting pipeline...")
    
    pipeline_start = datetime.utcnow()
    
    try:
        result = run_detection_pipeline_bytes(
            img_bytes, 
            generate_xai=generate_xai,
            language=language  # Pass language to pipeline
        )
        
        pipeline_time = (datetime.utcnow() - pipeline_start).total_seconds()
        print(f"   ✅ Pipeline completed in {pipeline_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ ML PIPELINE ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Disease detection failed: {str(e)}")

    # ================= STEP 7: Result Analysis =================
    print(f"\n📊 STEP 7: Result Analysis")
    print(f"   - Result keys: {list(result.keys())}")
    print(f"   - Is plant: {result.get('is_plant')}")
    
    if result.get('is_plant'):
        print(f"   - Plant type: {result.get('plant_type')}")
        print(f"   - Plant type confidence: {result.get('plant_type_confidence', 0)*100:.2f}%")
        print(f"   - Disease: {result.get('disease')}")
        print(f"   - Disease confidence: {result.get('disease_confidence', 0)*100:.2f}%")
        print(f"   - Severity: {result.get('severity')}")
        print(f"   - Severity score: {result.get('severity_score')}/3")
        print(f"   - Severity message: {result.get('severity_message')}")
        print(f"   - Recommendations: {len(result.get('recommendations', []))} items")
        print(f"   - XAI available: {result.get('xai_available', False)}")
        
        if result.get('gradcam_overlay'):
            overlay_size = len(result.get('gradcam_overlay', ''))
            print(f"   - Grad-CAM overlay size: {overlay_size:,} chars ({overlay_size/1024:.2f} KB)")
        else:
            print(f"   - Grad-CAM overlay: None")
    else:
        print(f"   - Not a plant part detected")
        print(f"   - Message: {result.get('message', 'N/A')}")

    # ================= STEP 8: Build Response =================
    print(f"\n📤 STEP 8: Building Response")
    
    response_data = {
        "status": "ok",
        "integrity": "passed",
        "result": result,
        "metadata": {
            "username": username,
            "device_id": device_id,
            "language": language,
            "xai_generated": result.get('xai_available', False),
            "processing_time_seconds": pipeline_time,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    
    print(f"   - Response status: {response_data['status']}")
    print(f"   - Integrity check: {response_data['integrity']}")
    print(f"   - Metadata included: Yes")
    
    # ================= Summary =================
    print(f"\n{'='*70}")
    print(f"✅ /predict ENDPOINT COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"   📊 Summary:")
    print(f"   - User: {username}")
    print(f"   - Device: {device_id}")
    print(f"   - Language: {language}")
    print(f"   - Image: {len(img_bytes)/1024:.2f} KB")
    print(f"   - Plant detected: {result.get('is_plant')}")
    if result.get('is_plant'):
        print(f"   - Disease: {result.get('disease')}")
        print(f"   - Confidence: {result.get('disease_confidence', 0)*100:.2f}%")
        print(f"   - Severity: {result.get('severity')} (score: {result.get('severity_score')}/3)")
    print(f"   - XAI: {'Generated' if result.get('xai_available') else 'Not generated'}")
    print(f"   - Total time: {pipeline_time:.2f}s")
    print(f"{'='*70}\n")

    return response_data



# ---------- MAIN ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
