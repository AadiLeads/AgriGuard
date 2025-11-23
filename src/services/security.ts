// src/services/security.ts
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import CryptoJS from "crypto-js";
import { KJUR, hextob64, KEYUTIL } from "jsrsasign";

const BASE_URL = "https://dwb3r7h0-8000.inc1.devtunnels.ms";

const PRIVATE_KEY_KEY = "privateKeyPem";
const PUBLIC_KEY_KEY = "publicKeyPem";
const DEVICE_ID_KEY = "device_id";

// ---------- Types ----------
export type PickedImage = {
  uri: string;
  base64?: string | null;
  mimeType?: string | null;
  fileName?: string | null;
};

// ---------- Helpers ----------
function bytesToHex(bytes: Uint8Array): string {
  let hex = "";
  for (let i = 0; i < bytes.length; i++) {
    hex += bytes[i].toString(16).padStart(2, "0");
  }
  return hex;
}

// Load raw image bytes from a file URI (Expo-compatible)
export async function getImageBytes(uri: string): Promise<Uint8Array> {
  const response = await fetch(uri);
  const buffer = await response.arrayBuffer();
  return new Uint8Array(buffer);
}

// Generate a unique device id per install and persist it
async function getDeviceId(): Promise<string> {
  const existing = await SecureStore.getItemAsync(DEVICE_ID_KEY);
  if (existing) return existing;

  const newId = `device-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  await SecureStore.setItemAsync(DEVICE_ID_KEY, newId);
  console.log("🆔 Generated new device_id:", newId);
  return newId;
}

// Generate RSA keypair on device (pure JS, 1024 bits for performance)
async function generateRsaKeypair(bits: number = 1024): Promise<{
  privateKeyPem: string;
  publicKeyPem: string;
}> {
  return new Promise((resolve, reject) => {
    try {
      console.log(`🧮 Starting RSA key generation (${bits} bits)…`);
      const kp = KEYUTIL.generateKeypair("RSA", bits);
      console.log("✅ Finished RSA key generation.");

      // Use PKCS8 private key – BEGIN PRIVATE KEY –
      const privateKeyPem = KEYUTIL.getPEM(kp.prvKeyObj, "PKCS8PRV");
      // Standard public key PEM – BEGIN PUBLIC KEY –
      const publicKeyPem = KEYUTIL.getPEM(kp.pubKeyObj);

      resolve({ privateKeyPem, publicKeyPem });
    } catch (e) {
      console.log("❌ RSA key generation error:", e);
      reject(e);
    }
  });
}

// Validate existing keys from SecureStore; if bad, regenerate
async function validateOrGenerateKeys(): Promise<{
  privateKeyPem: string;
  publicKeyPem: string;
}> {
  let privateKeyPem = await SecureStore.getItemAsync(PRIVATE_KEY_KEY);
  let publicKeyPem = await SecureStore.getItemAsync(PUBLIC_KEY_KEY);

  let needNew = false;

  if (!privateKeyPem || !publicKeyPem) {
    console.log("⚠️ Missing private or public key in SecureStore.");
    needNew = true;
  } else {
    try {
      // If this throws, the PEM is not a valid key
      KEYUTIL.getKey(privateKeyPem);
    } catch (e) {
      console.log("⚠️ Existing private key invalid/corrupted:", e);
      needNew = true;
    }
  }

  if (needNew) {
    console.log("🔑 Generating new RSA keypair for this device…");
    const { privateKeyPem: priv, publicKeyPem: pub } = await generateRsaKeypair(
      1024
    );
    privateKeyPem = priv;
    publicKeyPem = pub;
    await SecureStore.setItemAsync(PRIVATE_KEY_KEY, privateKeyPem);
    await SecureStore.setItemAsync(PUBLIC_KEY_KEY, publicKeyPem);
  }

  return { privateKeyPem: privateKeyPem!, publicKeyPem: publicKeyPem! };
}

// helper: call /register-device
async function registerDevice(deviceId: string, publicKeyPem: string) {
  const form = new FormData();
  form.append("device_id", deviceId);
  form.append("public_key", publicKeyPem);

  console.log("🌐 Calling backend (FORM):", `${BASE_URL}/register-device`, {
    deviceId,
  });

  const res = await fetch(`${BASE_URL}/register-device`, {
    method: "POST",
    body: form,
  });

  const text = await res.text();
  console.log("📥 /register-device status:", res.status);
  console.log("📥 /register-device body:", text);

  if (!res.ok) {
    throw new Error(
      `Device registration failed (status ${res.status}): ${text}`
    );
  }
}

// ------------------------------------------------------------
// 1) Setup keys + Register device (per device, FormData)
// ------------------------------------------------------------
export async function setupKeys() {
  const deviceId = await getDeviceId();

  const { privateKeyPem, publicKeyPem } = await validateOrGenerateKeys();

  console.log(
    "🔐 privateKeyPem prefix:",
    privateKeyPem.slice(0, 40).replace(/\n/g, " ")
  );

  await registerDevice(deviceId, publicKeyPem);
  console.log("✅ Device registered/re-registered with backend.");
}

// ------------------------------------------------------------
// 2) Image Picker (Expo Go compatible)
// ------------------------------------------------------------
export async function selectImage(): Promise<PickedImage | null> {
  const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
  if (status !== "granted") {
    alert("Permission to access gallery is required!");
    return null;
  }

  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ImagePicker.MediaTypeOptions.Images,
    base64: false, // we work from raw bytes instead
    quality: 1,
  });

  if (result.canceled || !result.assets || result.assets.length === 0) {
    return null;
  }

  const asset = result.assets[0];

  return {
    uri: asset.uri,
    base64: asset.base64 ?? null,
    mimeType: asset.mimeType ?? "image/jpeg",
    fileName: asset.fileName ?? "upload.jpg",
  };
}

// ------------------------------------------------------------
// 3) SHA256 hash of RAW BYTES (matches backend digest)
// ------------------------------------------------------------
export async function hashImage(imageBytes: Uint8Array): Promise<string> {
  const hex = bytesToHex(imageBytes);
  const wordArray = CryptoJS.enc.Hex.parse(hex);
  const hashHex = CryptoJS.SHA256(wordArray).toString(CryptoJS.enc.Hex);
  return hashHex;
}

// ------------------------------------------------------------
// 4) RSA SHA256 Sign over the hash (compatible with backend)
// ------------------------------------------------------------
export async function signHash(hashHex: string): Promise<string> {
  const privateKeyPem = await SecureStore.getItemAsync(PRIVATE_KEY_KEY);
  if (!privateKeyPem) {
    throw new Error("Private key not found. Call setupKeys() first.");
  }

  console.log(
    "🔐 privateKeyPem prefix in signHash:",
    privateKeyPem.slice(0, 40).replace(/\n/g, " ")
  );

  try {
    const sig = new KJUR.crypto.Signature({ alg: "SHA256withRSA" });
    sig.init(privateKeyPem); // throws if PEM invalid
    sig.updateHex(hashHex);

    const signatureHex = sig.sign();
    const signatureBase64 = hextob64(signatureHex); // backend expects base64
    return signatureBase64;
  } catch (e) {
    console.log("❌ signHash error, resetting keys:", e);

    // If we get here, keys are bad → wipe and force regeneration next run
    await SecureStore.deleteItemAsync(PRIVATE_KEY_KEY);
    await SecureStore.deleteItemAsync(PUBLIC_KEY_KEY);
    await SecureStore.deleteItemAsync(DEVICE_ID_KEY);

    throw new Error(
      "Crypto keys were corrupted and have been reset. Please try again."
    );
  }
}

// ------------------------------------------------------------
// 5) Upload Image + Signature + token + csrf to /predict
// ------------------------------------------------------------
export async function uploadImage(
  image: PickedImage,
  signature: string,
  token: string,
  csrf: string,
  generateXai: boolean = true,      // 🔥 Default to true for XAI
  language: string = "en"
): Promise<any> {
  console.log("\n🚀 uploadImage() called");
  console.log("   - generateXai:", generateXai);
  console.log("   - language:", language);
  console.log("   - token length:", token?.length);
  console.log("   - csrf length:", csrf?.length);

  if (!token || !csrf) {
    throw new Error("Missing auth token or csrf");
  }

  const deviceId = await getDeviceId();
  console.log("   - device_id:", deviceId);

  const form = new FormData();

  // Image
  form.append("image", {
    // @ts-ignore
    uri: image.uri,
    type: image.mimeType || "image/jpeg",
    name: image.fileName || "upload.jpg",
  } as any);

  // Auth & Security
  form.append("signature", signature);
  form.append("device_id", deviceId);
  form.append("token", token);
  form.append("csrf", csrf);

  // Options
  form.append("generate_xai", String(generateXai)); // "true" or "false"
  form.append("language", language);

  console.log("📤 Sending request to:", `${BASE_URL}/predict`);
  console.log("   - FormData keys:", [
    "image",
    "signature",
    "device_id",
    "token",
    "csrf",
    "generate_xai",
    "language",
  ]);

  try {
    const res = await fetch(`${BASE_URL}/predict`, {
      method: "POST",
      body: form,
    });

    const text = await res.text();
    console.log("📥 Response status:", res.status);
    console.log("📥 Response body preview:", text.substring(0, 200) + "...");

    if (!res.ok) {
      console.log("❌ Upload failed with status:", res.status);
      throw new Error(`Upload failed (${res.status}): ${text}`);
    }

    const jsonResponse = JSON.parse(text);
    console.log("✅ Upload successful");
    console.log("   - Status:", jsonResponse.status);
    console.log("   - Integrity:", jsonResponse.integrity);
    console.log("   - Is plant:", jsonResponse.result?.is_plant);
    console.log("   - XAI available:", jsonResponse.result?.xai_available);

    return jsonResponse;
  } catch (error: any) {
    console.log("❌ Upload error:", error.message);
    throw error;
  }
}
