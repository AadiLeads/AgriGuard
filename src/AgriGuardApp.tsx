import React, { useState } from "react";
import { View, Text, TouchableOpacity, ScrollView } from "react-native";
import { Home, Camera, User, Leaf } from "lucide-react-native";

import LoginScreen from "./screens/LoginScreen";
import RegisterScreen from "./screens/RegisterScreen";

import CameraScreen from "./screens/CameraScreen";
import HomeScreen from "./screens/HomeScreen";
import UploadScreen from "./screens/UploadScreen";
import ScanningScreen from "./screens/ScanningScreen";
import ResultsScreen from "./screens/ResultsScreen";
import AboutScreen from "./screens/AboutScreen";

import { styles } from "./styles/agriGuardStyles";

type Screen =
  | "login"
  | "signup"
  | "home"
  | "upload"
  | "camera"
  | "scanning"
  | "results"
  | "about";

export function AgriGuardApp() {
  const [analysisResult, setAnalysisResult] = useState<any | null>(null);
  const [currentScreen, setCurrentScreen] = useState<Screen>("login");
  const [scannedImage, setScannedImage] = useState<string | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // 🔥 Store image URI for translation (tokens come from useAuth hook)
  const [currentImageUri, setCurrentImageUri] = useState<string | null>(null);

  const navigateTo = (screen: Screen) => setCurrentScreen(screen);

  const handleLoginSuccess = (tokens: any) => {
    // tokens object contains { token, csrf } from useAuth
    setIsLoggedIn(true);
    setCurrentScreen("home");
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    setCurrentScreen("login");
  };

  // Login
  if (!isLoggedIn && currentScreen === "login") {
    return (
      <LoginScreen
        onLoginSuccess={handleLoginSuccess}
        goToRegister={() => navigateTo("signup")}
      />
    );
  }

  // Register
  if (!isLoggedIn && currentScreen === "signup") {
    return <RegisterScreen goToLogin={() => navigateTo("login")} />;
  }

  // Main app
  return (
    <View style={styles.container}>
      {/* Header (hidden on camera) */}
      {currentScreen !== "camera" && (
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View style={styles.logo}>
              <Leaf size={20} color="#fff" strokeWidth={2.5} />
            </View>
            <Text style={styles.logoText}>AgriGuard</Text>
          </View>
          <TouchableOpacity
            onPress={() => navigateTo("about")}
            style={styles.profileButton}
          >
            <User size={20} color="#000" />
          </TouchableOpacity>
        </View>
      )}

      {/* Content */}
      {currentScreen === "camera" ? (
        <CameraScreen
          onBack={() => navigateTo("upload")}
          onPhotoTaken={(uri) => {
            setScannedImage(uri);
            setCurrentImageUri(uri);  // 🔥 Store image URI for translation
            navigateTo("scanning");
          }}
        />
      ) : (
        <ScrollView style={styles.content} showsVerticalScrollIndicator={false}>
          {currentScreen === "home" && (
            <HomeScreen onScanClick={() => navigateTo("upload")} />
          )}

          {currentScreen === "upload" && (
            <UploadScreen
              onBack={() => navigateTo("home")}
              onCameraPress={() => navigateTo("camera")}
              onImageSelected={(uri) => {
                setScannedImage(uri);
                setCurrentImageUri(uri);  // 🔥 Store image URI for translation
                navigateTo("scanning");
              }}
            />
          )}

          {currentScreen === "scanning" && (
            <ScanningScreen
              image={scannedImage}
              onScanComplete={(result) => {
                setAnalysisResult(result);
                setCurrentScreen("results");
              }}
              onCancel={() => setCurrentScreen("home")}
            />
          )}

          {currentScreen === "results" && (
            <ResultsScreen
              data={analysisResult}
              onBack={() => navigateTo("home")}
              onScanAgain={() => navigateTo("upload")}
              imageUri={currentImageUri}  // 🔥 Pass image URI for translation
            />
          )}

          {currentScreen === "about" && (
            <AboutScreen
              onBack={() => navigateTo("home")}
              onLogout={handleLogout}
            />
          )}
        </ScrollView>
      )}

      {/* Bottom nav (hide on camera/upload/scanning) */}
      {currentScreen !== "scanning" &&
        currentScreen !== "upload" &&
        currentScreen !== "camera" && (
          <View style={styles.bottomNav}>
            <TouchableOpacity
              style={[
                styles.navButton,
                currentScreen === "home" && styles.navButtonActive,
              ]}
              onPress={() => navigateTo("home")}
            >
              <Home
                size={24}
                color={currentScreen === "home" ? "#22C55E" : "#999"}
              />
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.fabButton}
              onPress={() => navigateTo("upload")}
            >
              <Camera size={28} color="#fff" />
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.navButton,
                currentScreen === "about" && styles.navButtonActive,
              ]}
              onPress={() => navigateTo("about")}
            >
              <User
                size={24}
                color={currentScreen === "about" ? "#22C55E" : "#999"}
              />
            </TouchableOpacity>
          </View>
        )}
    </View>
  );
}