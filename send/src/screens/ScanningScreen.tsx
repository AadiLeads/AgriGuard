// src/screens/ScanningScreen.tsx
import React, { useEffect, useRef } from "react";
import { View, Text, Image, Animated, Alert } from "react-native";
import { styles } from "../styles/agriGuardStyles";
import {
  setupKeys,
  getImageBytes,
  hashImage,
  signHash,
  uploadImage,
} from "../services/security";
import { useAuth } from "../auth/useAuth";

type Props = {
  image: string | null;
  onScanComplete: (result: any) => void;
};

export default function ScanningScreen({ image, onScanComplete }: Props) {
  const scanAnim = useRef(new Animated.Value(0)).current;
  const { token, csrf } = useAuth();

  useEffect(() => {
    // Start scanning animation
    const anim = Animated.loop(
      Animated.timing(scanAnim, {
        toValue: 1,
        duration: 2200,
        useNativeDriver: true,
      })
    );
    anim.start();

    (async () => {
      try {
        if (!image) {
          throw new Error("No image provided");
        }
        if (!token || !csrf) {
          throw new Error("You are not logged in. Please login again.");
        }

        // 0. Ensure keys + device registration
        await setupKeys();

        // 1. Load raw bytes from URI
        const bytes = await getImageBytes(image);

        // 2. Hash bytes
        const hashHex = await hashImage(bytes);

        // 3. Sign hash
        const signature = await signHash(hashHex);

        // 4. Build image object for uploadImage
        const imgObj = {
          uri: image,
          mimeType: "image/jpeg",
          fileName: "scan.jpg",
        };

        // 5. Upload securely
        const result = await uploadImage(imgObj, signature, token, csrf, true);

        // 6. Hand result to parent
        onScanComplete(result);
      } catch (err: any) {
        console.log("❌ Scan pipeline error:", err);
        Alert.alert("Error", err.message || "Failed to scan image");
      }
    })();

    return () => {
      anim.stop();
    };
  }, [image, token, csrf]);

  const translateY = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, 250],
  });

  return (
    <View style={styles.scanningContainer}>
      <View style={styles.scanningImageContainer}>
        {image && (
          <Image
            source={{ uri: image }}
            style={styles.scanningImage}
            resizeMode="cover"
          />
        )}
        <Animated.View
          style={[styles.scanLine, { transform: [{ translateY }] }]}
        />
      </View>
      <Text style={styles.scanningText}>Analyzing  Structure...</Text>
      <Text style={styles.scanningSubtext}>Checking for symptoms</Text>
    </View>
  );
}
