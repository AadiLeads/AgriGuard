import React from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import * as ImagePicker from "expo-image-picker";
import { X, Leaf, Camera, ImageIcon } from "lucide-react-native";
import { styles } from "../styles/agriGuardStyles";

type Props = {
  onBack: () => void;
  onCameraPress: () => void;
  onImageSelected: (uri: string) => void;
};

export default function UploadScreen({
  onBack,
  onCameraPress,
  onImageSelected,
}: Props) {
  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        onImageSelected(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert("Error", "Failed to pick image");
      console.error(error);
    }
  };

  return (
    <View style={styles.screen}>
      <View style={styles.uploadHeader}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <X size={20} color="#000" />
        </TouchableOpacity>
        <Text style={styles.uploadTitle}>New Scan</Text>
        <View style={{ width: 40 }} />
      </View>

      <View style={styles.uploadContent}>
        <View style={styles.uploadIconContainer}>
          <Leaf size={48} color="#22C55E" />
        </View>
        <Text style={styles.uploadMainTitle}>Upload a Photo</Text>
        <Text style={styles.uploadSubtitle}>
          Take a clear picture of the affected leaf for accurate diagnosis.
        </Text>

        <TouchableOpacity
          style={styles.uploadButtonPrimary}
          onPress={onCameraPress}
        >
          <Camera size={24} color="#fff" />
          <Text style={styles.uploadButtonPrimaryText}>Take Photo</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.uploadButtonSecondary}
          onPress={pickImage}
        >
          <ImageIcon size={24} color="#000" />
          <Text style={styles.uploadButtonSecondaryText}>
            Upload from Gallery
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
