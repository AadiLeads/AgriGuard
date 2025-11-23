// src/screens/ResultsScreen.tsx
import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  Alert,
  ScrollView,
  Modal,
  ActivityIndicator,
} from "react-native";
import { X, Upload, Leaf, Languages } from "lucide-react-native";
import * as Print from "expo-print";
import * as Sharing from "expo-sharing";
import { styles } from "../styles/agriGuardStyles";
import {
  getImageBytes,
  hashImage,
  signHash,
  uploadImage,
} from "../services/security";
import { useAuth } from "../auth/useAuth"; // 🔥 Use auth hook

// 🔥 Updated to match backend response structure
type BackendResult = {
  status: string;
  integrity: string;
  result?: {
    is_plant?: boolean;
    plant_type?: string | null;
    plant_type_translated?: string | null;
    plant_type_confidence?: number | null;
    disease?: string | null;
    disease_translated?: string | null;
    disease_confidence?: number | null;
    recommendations?: string[];
    severity?: string | null;
    severity_translated?: string | null;
    severity_score?: number | null;
    severity_message?: string | null;
    severity_message_translated?: string | null;
    confidence_percent?: number | null;
    xai_available?: boolean;
    gradcam_overlay?: string | null;
    language?: string;
  };
  metadata?: {
    username?: string;
    device_id?: string;
    language?: string;
    xai_generated?: boolean;
    processing_time_seconds?: number;
    timestamp?: string;
  };
};

type Props = {
  onBack: () => void;
  onScanAgain: () => void;
  data: BackendResult | null;
  imageUri?: string | null;
};

// Language options matching backend
const LANGUAGES = [
  { code: "en", name: "English", flag: "🇬🇧" },
  { code: "es", name: "Spanish", flag: "🇪🇸" },
  { code: "fr", name: "French", flag: "🇫🇷" },
  { code: "de", name: "German", flag: "🇩🇪" },
  { code: "hi", name: "Hindi", flag: "🇮🇳" },
  { code: "zh", name: "Chinese", flag: "🇨🇳" },
  { code: "ar", name: "Arabic", flag: "🇸🇦" },
  { code: "pt", name: "Portuguese", flag: "🇵🇹" },
  { code: "ru", name: "Russian", flag: "🇷🇺" },
  { code: "ja", name: "Japanese", flag: "🇯🇵" },
];

export default function ResultsScreen({
  onBack,
  onScanAgain,
  data,
  imageUri,
}: Props) {
  const { token, csrf } = useAuth(); // 🔥 Get tokens from auth hook

  const [showLanguageModal, setShowLanguageModal] = useState(false);
  const [translating, setTranslating] = useState(false);
  const [translatedData, setTranslatedData] = useState<BackendResult | null>(
    data
  );
  const [currentLanguage, setCurrentLanguage] = useState(
    data?.metadata?.language || data?.result?.language || "en"
  );

  // Use translated data if available, otherwise original
  const displayData = translatedData || data;

  // Handle translation
  const handleTranslate = async (languageCode: string) => {
    if (!imageUri || !token || !csrf) {
      Alert.alert(
        "Error",
        "Missing required data for translation. Please scan again."
      );
      return;
    }

    if (languageCode === currentLanguage) {
      setShowLanguageModal(false);
      return;
    }

    setTranslating(true);
    setShowLanguageModal(false);

    try {
      console.log(`🌍 Translating to ${languageCode}...`);

      // Read image bytes
      const imageBytes = await getImageBytes(imageUri);

      // Hash image
      const hashHex = await hashImage(imageBytes);

      // Sign hash
      const signature = await signHash(hashHex);

      // Re-upload with new language
      const response = await uploadImage(
        { uri: imageUri, mimeType: "image/jpeg", fileName: "upload.jpg" },
        signature,
        token,
        csrf,
        true, // generate_xai
        languageCode
      );

      console.log("✅ Translation successful");
      setTranslatedData(response);
      setCurrentLanguage(languageCode);

      Alert.alert(
        "Success",
        `Results translated to ${
          LANGUAGES.find((l) => l.code === languageCode)?.name
        }`
      );
    } catch (error: any) {
      console.log("❌ Translation error:", error);
      Alert.alert(
        "Translation Error",
        error?.message || "Failed to translate results. Please try again."
      );
    } finally {
      setTranslating(false);
    }
  };

  // Handle no data case
  if (!displayData) {
    return (
      <View style={styles.screen}>
        <View style={styles.resultsHeader}>
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <X size={20} color="#000" />
          </TouchableOpacity>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          <View style={styles.backButton} />
        </View>

        <Text style={{ marginTop: 24, fontSize: 16, textAlign: "center" }}>
          No analysis result available. Please scan again.
        </Text>

        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={styles.actionButtonSecondary}
            onPress={onScanAgain}
          >
            <Text style={styles.actionButtonSecondaryText}>Scan Again</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Extract data from response
  const integrityOk = displayData.integrity === "passed";
  const result = displayData.result ?? {};
  const metadata = displayData.metadata ?? {};

  // 🔥 Handle case where plant is not detected
  if (!result.is_plant) {
    return (
      <View style={styles.screen}>
        <View style={styles.resultsHeader}>
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <X size={20} color="#000" />
          </TouchableOpacity>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          <View style={styles.backButton} />
        </View>

        <View style={styles.cardContainer}>
          <View style={styles.resultCard}>
            <View style={styles.resultCardHeader}>
              <Leaf size={80} color="#F97316" />
            </View>
            <View style={styles.resultCardContent}>
              <Text style={styles.resultDisease}>Not a Plant Part</Text>
              <Text style={[styles.resultDescription, { marginTop: 8 }]}>
                The image does not appear to contain a plant part. Please take a
                clearer photo of a leaf or fruit.
              </Text>
            </View>
          </View>
        </View>

        <View style={styles.actionButtons}>
          <TouchableOpacity
            style={styles.actionButtonSecondary}
            onPress={onScanAgain}
          >
            <Text style={styles.actionButtonSecondaryText}>Scan Again</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Parse disease info (use translated versions if available)
  const diseaseName = (
    result.disease_translated ||
    result.disease ||
    "No disease detected"
  ).replace(/_/g, " ");

  const diseaseConf =
    result.confidence_percent ??
    Math.round((result.disease_confidence ?? 0) * 100);

  const plantType =
    result.plant_type_translated || result.plant_type || "Unknown";
  const plantTypeConf = Math.round((result.plant_type_confidence ?? 0) * 100);
  const recommendations = result.recommendations ?? [];

  // Severity info (use translated versions if available)
  const severity = result.severity_translated || result.severity || "Unknown";
  const severityScore = result.severity_score ?? 0;
  const severityMessage =
    result.severity_message_translated ||
    result.severity_message ||
    "No severity information available.";

  // Severity colors
  const severityColorMap: Record<string, { bg: string; text: string }> = {
    Severe: { bg: "#FEE2E2", text: "#DC2626" },
    Moderate: { bg: "#FEF3C7", text: "#D97706" },
    Mild: { bg: "#ECFCCB", text: "#65A30D" },
    Healthy: { bg: "#DCFCE7", text: "#16A34A" },
    "Weak Detection": { bg: "#E5E7EB", text: "#4B5563" },
    Unknown: { bg: "#E5E7EB", text: "#4B5563" },
  };

  // Try to match severity (case-insensitive, partial match)
  let severityColors = severityColorMap["Unknown"];
  for (const key in severityColorMap) {
    if (severity.toLowerCase().includes(key.toLowerCase())) {
      severityColors = severityColorMap[key];
      break;
    }
  }

  // Grad-CAM
  const gradcamOverlay = result.gradcam_overlay;
  const xaiAvailable = !!result.xai_available && !!gradcamOverlay;

  // Get current language name
  const currentLangName =
    LANGUAGES.find((l) => l.code === currentLanguage)?.name || "English";

  // Generate PDF report
  const handleSaveReport = async () => {
    try {
      const now = new Date().toLocaleString();
      const recHtml =
        recommendations.length > 0
          ? "<ul>" + recommendations.map(r => "<li>" + r + "</li>").join("") + "</ul>"
          : "<p>No specific recommendations available.</p>";

      const gradcamHtml = xaiAvailable
        ? `
          <h3 style="margin-top:24px;">Model Focus (Grad-CAM)</h3>
          <p style="font-size:12px;color:#4B5563;">
            Highlighted areas show where the model focused to make this decision.
          </p>
          <img
            src="data:image/jpeg;base64,${gradcamOverlay}"
            style="width:100%;max-width:400px;border-radius:12px;border:1px solid #e5e7eb;"
          />
        `
        : "";
      const metadataHtml = metadata.processing_time_seconds
        ? `<p style="font-size:11px;color:#9CA3AF;">Processing time: ${metadata.processing_time_seconds.toFixed(
            2
          )}s</p>`
        : "";

      const html = `
        <html>
          <head>
            <meta charset="utf-8" />
            <title>AgriGuard Report</title>
          </head>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; color: #111827;">
            <h1 style="color:#16A34A;margin-bottom:4px;">🌾 AgriGuard Analysis Report</h1>
            <p style="font-size:12px;color:#6B7280;margin-top:0;">Generated on ${now}</p>
            <p style="font-size:11px;color:#9CA3AF;">Language: ${currentLangName}</p>
            ${metadataHtml}
            <hr style="margin:16px 0;border:none;border-top:1px solid #e5e7eb;" />

            <h2 style="margin-bottom:4px;">Summary</h2>
            <p><strong>Disease:</strong> ${diseaseName}</p>
            <p><strong>Plant type:</strong> ${plantType}</p>
            <p><strong>Disease confidence:</strong> ${diseaseConf}%</p>
            <p><strong>Plant type confidence:</strong> ${plantTypeConf}%</p>
            <p><strong>Severity:</strong> ${severity} (${severityScore}/3)</p>
            <p style="font-size:13px;color:#4B5563;">${severityMessage}</p>

            <h2 style="margin-top:24px;margin-bottom:8px;">Recommendations</h2>
            ${recHtml}

            ${gradcamHtml}
            
            <hr style="margin:24px 0;border:none;border-top:1px solid #e5e7eb;" />
            <p style="font-size:10px;color:#9CA3AF;text-align:center;">
              Generated by AgriGuard • Plant Disease Detection System
            </p>
          </body>
        </html>
      `;

      console.log("📝 Generating PDF report...");
      const { uri } = await Print.printToFileAsync({ html });
      console.log("✅ PDF generated at:", uri);

      const canShare = await Sharing.isAvailableAsync();
      if (!canShare) {
        Alert.alert(`Report saved, PDF generated at:\n${uri}`);
        return;
      }

      await Sharing.shareAsync(uri, {
        mimeType: "application/pdf",
        dialogTitle: "Share AgriGuard report",
      });
    } catch (err: any) {
      console.log("❌ Error generating report:", err);
      Alert.alert(
        "Error",
        err?.message || "Failed to generate or share the report."
      );
    }
  };

  return (
    <ScrollView style={styles.screen}>
      <View style={styles.resultsHeader}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <X size={20} color="#000" />
        </TouchableOpacity>
        <Text style={styles.resultsTitle}>Analysis Results</Text>
        <TouchableOpacity
          style={styles.backButton}
          onPress={() => setShowLanguageModal(true)}
          disabled={translating}
        >
          {translating ? (
            <ActivityIndicator size="small" color="#16A34A" />
          ) : (
            <Languages size={20} color="#16A34A" />
          )}
        </TouchableOpacity>
      </View>

      {/* Current Language Indicator */}
      {currentLanguage !== "en" && (
        <View
          style={{
            marginHorizontal: 16,
            marginTop: 8,
            paddingHorizontal: 12,
            paddingVertical: 6,
            backgroundColor: "#EFF6FF",
            borderRadius: 8,
            alignSelf: "flex-start",
          }}
        >
          <Text style={{ fontSize: 12, color: "#1D4ED8" }}>
            {LANGUAGES.find((l) => l.code === currentLanguage)?.flag}{" "}
            {currentLangName}
          </Text>
        </View>
      )}

      <View style={styles.cardContainer}>
        {/* Main Result Card */}
        <View style={styles.resultCard}>
          <View style={styles.resultCardHeader}>
            <Leaf size={80} color={integrityOk ? "#22C55E" : "#F97316"} />
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceText}>
                {diseaseConf}% Confidence
              </Text>
            </View>
          </View>

          <View style={styles.resultCardContent}>
            <View style={styles.resultCardTitleRow}>
              <View style={{ flex: 1 }}>
                <Text style={styles.resultDisease}>{diseaseName}</Text>
                <Text style={styles.resultScientific}>
                  Plant type: {plantType}
                </Text>
              </View>

              <View
                style={[
                  styles.criticalBadge,
                  !integrityOk && { backgroundColor: "#FEE2E2" },
                ]}
              >
                <Text
                  style={[
                    styles.criticalText,
                    !integrityOk && { color: "#DC2626" },
                  ]}
                >
                  {integrityOk ? "VERIFIED" : "CHECK"}
                </Text>
              </View>
            </View>

            {/* Confidence Chips */}
            <View
              style={{
                flexDirection: "row",
                marginTop: 12,
                gap: 8,
                flexWrap: "wrap",
              }}
            >
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 999,
                  backgroundColor: "#ECFDF3",
                }}
              >
                <Text style={{ fontSize: 12, color: "#15803D" }}>
                  Disease: {diseaseConf}%
                </Text>
              </View>
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 999,
                  backgroundColor: "#EFF6FF",
                }}
              >
                <Text style={{ fontSize: 12, color: "#1D4ED8" }}>
                  Type: {plantTypeConf}%
                </Text>
              </View>
            </View>

            {/* Severity Badge */}
            <View style={{ marginTop: 12 }}>
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 8,
                  borderRadius: 999,
                  backgroundColor: severityColors.bg,
                  alignSelf: "flex-start",
                }}
              >
                <Text
                  style={{
                    fontSize: 13,
                    fontWeight: "700",
                    color: severityColors.text,
                  }}
                >
                  Severity: {severity} ({severityScore}/3)
                </Text>
              </View>
              <Text style={{ marginTop: 8, fontSize: 13, color: "#4B5563" }}>
                {severityMessage}
              </Text>
            </View>

            {/* Recommendations */}
            <Text
              style={[
                styles.resultDescription,
                { marginTop: 16, fontWeight: "600" },
              ]}
            >
              Recommended Actions:
            </Text>

            {recommendations.length > 0 ? (
              <View style={{ marginTop: 8 }}>
                {recommendations.map((rec, idx) => (
                  <View
                    key={idx}
                    style={{ flexDirection: "row", marginTop: 6 }}
                  >
                    <Text style={{ marginRight: 6, color: "#16A34A" }}>•</Text>
                    <Text style={{ flex: 1, color: "#4B5563", fontSize: 14 }}>
                      {rec}
                    </Text>
                  </View>
                ))}
              </View>
            ) : (
              <Text style={{ marginTop: 8, color: "#6B7280", fontSize: 14 }}>
                No specific recommendations available.
              </Text>
            )}
          </View>
        </View>

        {/* Grad-CAM Card */}
        {xaiAvailable && (
          <View
            style={{
              marginTop: 16,
              backgroundColor: "#FFFFFF",
              borderRadius: 16,
              borderWidth: 1,
              borderColor: "#E5E7EB",
              padding: 16,
              shadowColor: "#000",
              shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.05,
              shadowRadius: 8,
              elevation: 2,
            }}
          >
            <Text
              style={{
                fontSize: 15,
                fontWeight: "700",
                marginBottom: 6,
                color: "#111827",
              }}
            >
              🔍 Model Focus (Explainable AI)
            </Text>
            <Text
              style={{
                fontSize: 12,
                color: "#6B7280",
                marginBottom: 12,
                lineHeight: 18,
              }}
            >
              The highlighted areas show where the AI model focused its
              attention when making this diagnosis.
            </Text>

            <Image
              source={{
                uri: `data:image/jpeg;base64,${gradcamOverlay}`,
              }}
              style={{
                width: "100%",
                height: 240,
                borderRadius: 12,
                backgroundColor: "#F3F4F6",
              }}
              resizeMode="contain"
            />
          </View>
        )}

        {/* Metadata Card (if available) */}
        {metadata.processing_time_seconds && (
          <View
            style={{
              marginTop: 12,
              padding: 12,
              backgroundColor: "#F9FAFB",
              borderRadius: 12,
            }}
          >
            <Text style={{ fontSize: 11, color: "#6B7280" }}>
              ⏱ Processed in {metadata.processing_time_seconds.toFixed(2)}s
              {metadata.timestamp &&
                ` • ${new Date(metadata.timestamp).toLocaleString()}`}
            </Text>
          </View>
        )}
      </View>

      {/* Action Buttons */}
      <View style={[styles.actionButtons, { marginBottom: 24 }]}>
        <TouchableOpacity
          style={styles.actionButtonSecondary}
          onPress={onScanAgain}
        >
          <Text style={styles.actionButtonSecondaryText}>Scan Again</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.actionButtonPrimary}
          onPress={handleSaveReport}
        >
          <Text style={styles.actionButtonPrimaryText}>Save Report</Text>
        </TouchableOpacity>
      </View>

      {/* Language Selection Modal */}
      <Modal
        visible={showLanguageModal}
        transparent
        animationType="slide"
        onRequestClose={() => setShowLanguageModal(false)}
      >
        <View
          style={{
            flex: 1,
            backgroundColor: "rgba(0,0,0,0.5)",
            justifyContent: "flex-end",
          }}
        >
          <View
            style={{
              backgroundColor: "#FFFFFF",
              borderTopLeftRadius: 24,
              borderTopRightRadius: 24,
              paddingTop: 20,
              paddingBottom: 40,
              paddingHorizontal: 16,
              maxHeight: "70%",
            }}
          >
            {/* Header */}
            <View
              style={{
                flexDirection: "row",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 16,
              }}
            >
              <Text
                style={{ fontSize: 18, fontWeight: "700", color: "#111827" }}
              >
                Select Language
              </Text>
              <TouchableOpacity onPress={() => setShowLanguageModal(false)}>
                <X size={24} color="#6B7280" />
              </TouchableOpacity>
            </View>

            {/* Language List */}
            <ScrollView showsVerticalScrollIndicator={false}>
              {LANGUAGES.map((lang) => (
                <TouchableOpacity
                  key={lang.code}
                  style={{
                    flexDirection: "row",
                    alignItems: "center",
                    paddingVertical: 14,
                    paddingHorizontal: 16,
                    backgroundColor:
                      currentLanguage === lang.code ? "#ECFDF5" : "#FFFFFF",
                    borderRadius: 12,
                    marginBottom: 8,
                    borderWidth: 1,
                    borderColor:
                      currentLanguage === lang.code ? "#16A34A" : "#E5E7EB",
                  }}
                  onPress={() => handleTranslate(lang.code)}
                >
                  <Text style={{ fontSize: 24, marginRight: 12 }}>
                    {lang.flag}
                  </Text>
                  <Text
                    style={{
                      flex: 1,
                      fontSize: 16,
                      color: "#111827",
                      fontWeight: currentLanguage === lang.code ? "600" : "400",
                    }}
                  >
                    {lang.name}
                  </Text>
                  {currentLanguage === lang.code && (
                    <View
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: "#16A34A",
                      }}
                    />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}
