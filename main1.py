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
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import (
    preprocess_input as resnet50_preprocess,
)
import warnings

from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

warnings.filterwarnings("ignore")

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

fruit_model = load_model(FRUIT_MODEL_PATH)

# ------------------------------
# DISEASE RECOMMENDATIONS DB
# (unchanged from your code)
# ------------------------------

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


def predict_leaf_disease(image_path):
    """Predict disease for leaf images"""
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img)
    xb = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        preds = leaf_model(xb)
        probs = torch.nn.functional.softmax(preds, dim=1)
        conf, idx = torch.max(probs, dim=1)

    predicted_class = leaf_class_names[idx.item()]
    confidence = conf.item()

    return predicted_class, confidence


def predict_fruit_disease(image_path):
    """Predict disease for fruit images"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet50_preprocess(img_array)

    pred = fruit_model.predict(img_array, verbose=0)[0]
    label_idx = np.argmax(pred)
    confidence = float(pred[label_idx])

    predicted_class = fruit_class_names[label_idx]
    return predicted_class, confidence


def get_recommendations(disease_name):
    """Get treatment recommendations for a disease"""
    for key in disease_recommendations:
        if key.lower() in disease_name.lower():
            return disease_recommendations[key]

    return [
        " Consult with a local agricultural extension officer",
        " Remove and destroy infected plant parts",
        " Monitor plant regularly and maintain good hygiene",
    ]


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


def run_detection_pipeline_bytes(image_bytes: bytes):
    """
    Backend version of your full detection system.
    Accepts raw image bytes, returns structured JSON.
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    # Step 1: Check plant part
    is_plant = is_plant_part(img_array)

    if not is_plant:
        return {
            "is_plant": False,
            "message": "NOT A PLANT PART DETECTED",
        }

    # Step 2: classify plant type
    plant_type, type_confidence = classify_plant_type(img_array)

    # Step 3: disease prediction uses image_path, so write temp file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(image_bytes)
        tmp.flush()
        temp_path = tmp.name

        if plant_type == "LEAF":
            prediction, confidence = predict_leaf_disease(temp_path)
        else:
            prediction, confidence = predict_fruit_disease(temp_path)

    recs = get_recommendations(prediction)

    return {
        "is_plant": True,
        "plant_type": plant_type,
        "plant_type_confidence": float(type_confidence),
        "disease": prediction,
        "disease_confidence": float(confidence),
        "recommendations": recs,
    }


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

@app.post("/predict")
async def predict(
    request: Request,
    image: UploadFile = File(...),
    signature: str = Form(...),
    device_id: str = Form(...),
    token: str = Form(...),
    csrf: str = Form(...),
):
    # 1) JWT
    payload = verify_jwt(token)
    username = payload["sub"]

    # 2) CSRF
    verify_csrf(username, csrf)

    # 3) Device & public key
    if device_id not in devices_db:
        raise HTTPException(status_code=400, detail="Unknown device")
    public_key_pem = devices_db[device_id]

    # 4) Read image
    img_bytes = await image.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    # 5) Verify signature (base64 -> bytes)
    try:
        raw_sig = base64.b64decode(signature)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid signature encoding")

    if not verify_image_signature(img_bytes, raw_sig, public_key_pem):
        raise HTTPException(status_code=400, detail="Tampered or invalid image!")

    # 6) Run full ML pipeline
    result = run_detection_pipeline_bytes(img_bytes)

    return {
        "status": "ok",
        "integrity": "passed",
        "result": result,
    }
    
    


# ---------- MAIN ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
