import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { Leaf, X } from "lucide-react-native";
import { styles } from "../styles/agriGuardStyles";

type Props = {
  onBack: () => void;
  onPhotoTaken: (uri: string) => void;
};

export default function CameraScreen({ onBack, onPhotoTaken }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraRef, setCameraRef] = useState<CameraView | null>(null);

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, []);

  if (!permission) {
    return (
      <View style={styles.cameraLoading}>
        <Text>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.cameraPermission}>
        <Leaf size={64} color="#22C55E" />
        <Text style={styles.permissionTitle}>Camera Access Required</Text>
        <Text style={styles.permissionText}>
          AgriGuard needs camera access to scan your plants for diseases.
        </Text>
        <TouchableOpacity
          style={styles.permissionButton}
          onPress={requestPermission}
        >
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.permissionBackButton} onPress={onBack}>
          <Text style={styles.permissionBackText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef) {
      try {
        const photo = await cameraRef.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        onPhotoTaken(photo.uri);
      } catch (error) {
        Alert.alert("Error", "Failed to take picture");
        console.error(error);
      }
    }
  };

  return (
    <View style={styles.cameraContainer}>
      <CameraView
        style={styles.camera}
        facing="back"
        ref={(ref) => setCameraRef(ref)}
      >
        <View style={styles.cameraOverlay}>
          {/* Top bar */}
          <View style={styles.cameraTopBar}>
            <TouchableOpacity
              onPress={onBack}
              style={styles.cameraBackButton}
            >
              <X size={24} color="#fff" />
            </TouchableOpacity>
            <Text style={styles.cameraTitle}>Scan Plant</Text>
            <View style={{ width: 40 }} />
          </View>

          {/* Center guide */}
          <View style={styles.cameraGuide}>
            <View style={styles.guideBox}>
              <View style={[styles.guideCorner, styles.guideTopLeft]} />
              <View style={[styles.guideCorner, styles.guideTopRight]} />
              <View style={[styles.guideCorner, styles.guideBottomLeft]} />
              <View style={[styles.guideCorner, styles.guideBottomRight]} />
            </View>
            <Text style={styles.guideText}>
              Position the leaf within the frame
            </Text>
          </View>

          {/* Bottom controls */}
          <View style={styles.cameraBottomBar}>
            <View style={{ width: 60 }} />
            <TouchableOpacity
              style={styles.captureButton}
              onPress={takePicture}
            >
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>
            <View style={{ width: 60 }} />
          </View>
        </View>
      </CameraView>
    </View>
  );
}
