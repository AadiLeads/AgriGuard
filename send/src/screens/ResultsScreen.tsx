// src/screens/ResultsScreen.tsx
import React from "react";
import { View, Text, TouchableOpacity, Image, Alert } from "react-native";
import { X, Upload, Leaf } from "lucide-react-native";
import * as Print from "expo-print";
import * as Sharing from "expo-sharing";
import { styles } from "../styles/agriGuardStyles";

type BackendResult = {
  status: string;
  integrity: string;
  result?: {
    is_plant?: boolean;
    plant_type?: string | null;
    plant_type_confidence?: number | null;
    disease?: string | null;
    disease_confidence?: number | null;
    recommendations?: string[];

    // 🔥 new fields from backend
    severity?: string | null;
    severity_score?: number | null;
    severity_message?: string | null;
    confidence_percent?: number | null;
    xai_available?: boolean;
    gradcam_overlay?: string | null;
  };
};

type Props = {
  onBack: () => void;
  onScanAgain: () => void;
  data: BackendResult | null;
};

export default function ResultsScreen({ onBack, onScanAgain, data }: Props) {
  if (!data) {
    return (
      <View style={styles.screen}>
        <View style={styles.resultsHeader}>
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <X size={20} color="#000" />
          </TouchableOpacity>
          <Text style={styles.resultsTitle}>Analysis Results</Text>
          <View style={styles.backButton} />
        </View>

        <Text style={{ marginTop: 24, fontSize: 16 }}>
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

  const integrityOk = data.integrity === "passed";
  const result = data.result ?? {};

  const diseaseName = (result.disease || "No disease detected").replace(
    /_/g,
    " "
  );
  const diseaseConf = Math.round((result.disease_confidence ?? 0) * 100);
  const plantType = result.plant_type || "Unknown";
  const plantTypeConf = Math.round(
    (result.plant_type_confidence ?? 0) * 100
  );
  const recommendations = result.recommendations ?? [];

  // 🔥 severity-related fields
  const severity = result.severity || "Unknown";
  const severityScore = result.severity_score ?? 0;
  const severityMessage =
    result.severity_message || "No severity information available.";
  const confidencePercent =
    result.confidence_percent ?? diseaseConf; // fallback if not present

  // Simple color mapping for severity badge
  const severityColorMap: Record<string, { bg: string; text: string }> = {
    Severe: { bg: "#FEE2E2", text: "#DC2626" },
    Moderate: { bg: "#FEF3C7", text: "#D97706" },
    Mild: { bg: "#ECFCCB", text: "#65A30D" },
    Healthy: { bg: "#DCFCE7", text: "#16A34A" },
    "Weak Detection": { bg: "#E5E7EB", text: "#4B5563" },
    Unknown: { bg: "#E5E7EB", text: "#4B5563" },
  };

  const severityColors =
    severityColorMap[severity] || severityColorMap["Unknown"];

  // 🔥 Grad-CAM overlay
  const gradcamOverlay = result.gradcam_overlay;
  const xaiAvailable = !!result.xai_available && !!gradcamOverlay;

  // 🔥 Generate PDF & open share dialog
  const handleSaveReport = async () => {
    try {
      // Simple HTML report – you can style this more later
      const now = new Date().toLocaleString();
      const recHtml =
        recommendations.length > 0
          ? `<ul>${recommendations
              .map((r) => `<li>${r}</li>`)
              .join("")}</ul>`
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

      const html = `
        <html>
          <head>
            <meta charset="utf-8" />
            <title>AgriGuard Report</title>
          </head>
          <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; color: #111827;">
            <h1 style="color:#16A34A;margin-bottom:4px;">🌾 AgriGuard Analysis Report</h1>
            <p style="font-size:12px;color:#6B7280;margin-top:0;">Generated on ${now}</p>
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
          </body>
        </html>
      `;

      console.log("📝 Generating PDF report...");
      const { uri } = await Print.printToFileAsync({ html });
      console.log("✅ PDF generated at:", uri);

      const canShare = await Sharing.isAvailableAsync();
      if (!canShare) {
        Alert.alert("Report saved", `PDF generated at:\n${uri}`);
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
    <View style={styles.screen}>
      <View style={styles.resultsHeader}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <X size={20} color="#000" />
        </TouchableOpacity>
        <Text style={styles.resultsTitle}>Analysis Results</Text>
        <TouchableOpacity style={styles.backButton}>
          <Upload size={20} color="#000" />
        </TouchableOpacity>
      </View>

      <View style={styles.cardContainer}>
        <View style={styles.resultCard}>
          <View style={styles.resultCardHeader}>
            <Leaf size={80} color={integrityOk ? "#22C55E" : "#F97316"} />
            <View style={styles.confidenceBadge}>
              <Text style={styles.confidenceText}>
                {confidencePercent}% Confidence
              </Text>
            </View>
          </View>

          <View style={styles.resultCardContent}>
            <View style={styles.resultCardTitleRow}>
              <View>
                <Text style={styles.resultDisease}>{diseaseName}</Text>
                <Text style={styles.resultScientific}>
                  Plant type: {plantType}
                </Text>
              </View>

              <View
                style={[
                  styles.criticalBadge,
                  !result.is_plant && { backgroundColor: "#E5E7EB" },
                ]}
              >
                <Text
                  style={[
                    styles.criticalText,
                    !result.is_plant && { color: "#4B5563" },
                  ]}
                >
                  {integrityOk ? "INTEGRITY OK" : "CHECK IMAGE"}
                </Text>
              </View>
            </View>

            {/* Chips row */}
            <View style={{ flexDirection: "row", marginTop: 12, gap: 8 }}>
              <View
                style={{
                  paddingHorizontal: 12,
                  paddingVertical: 6,
                  borderRadius: 999,
                  backgroundColor: "#ECFDF3",
                }}
              >
                <Text style={{ fontSize: 12, color: "#15803D" }}>
                  Disease confidence: {diseaseConf}%
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
                  Type confidence: {plantTypeConf}%
                </Text>
              </View>
            </View>

            {/* 🔥 Severity badge + text */}
            <View
              style={{
                marginTop: 12,
                flexDirection: "row",
                alignItems: "center",
                gap: 8,
              }}
            >
              <View
                style={{
                  paddingHorizontal: 10,
                  paddingVertical: 6,
                  borderRadius: 999,
                  backgroundColor: severityColors.bg,
                }}
              >
                <Text
                  style={{
                    fontSize: 12,
                    fontWeight: "700",
                    color: severityColors.text,
                  }}
                >
                  Severity: {severity}
                  {typeof severityScore === "number" ? ` (${severityScore}/3)` : ""}
                </Text>
              </View>
            </View>

            <Text
              style={{
                marginTop: 6,
                fontSize: 13,
                color: "#4B5563",
              }}
            >
              {severityMessage}
            </Text>

            {/* Recommendations */}
            <Text style={[styles.resultDescription, { marginTop: 14 }]}>
              Recommended actions based on the detected disease:
            </Text>

            {recommendations.length > 0 ? (
              <View style={{ marginTop: 8 }}>
                {recommendations.map((rec, idx) => (
                  <View
                    key={idx}
                    style={{ flexDirection: "row", marginTop: 4 }}
                  >
                    <Text style={{ marginRight: 6 }}>•</Text>
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

        {/* 🔥 Grad-CAM overlay card */}
        {xaiAvailable && (
          <View
            style={{
              marginTop: 12,
              backgroundColor: "#FFFFFF",
              borderRadius: 16,
              borderWidth: 1,
              borderColor: "#E5E7EB",
              padding: 16,
            }}
          >
            <Text
              style={{
                fontSize: 14,
                fontWeight: "700",
                marginBottom: 8,
                color: "#111827",
              }}
            >
              Model Focus (Grad-CAM)
            </Text>
            <Text
              style={{
                fontSize: 12,
                color: "#6B7280",
                marginBottom: 8,
              }}
            >
              Highlighted areas show where the model looked to make this
              decision.
            </Text>

            <Image
              source={{
                uri: `data:image/jpeg;base64,${gradcamOverlay}`,
              }}
              style={{
                width: "100%",
                height: 220,
                borderRadius: 12,
                backgroundColor: "#000",
              }}
              resizeMode="cover"
            />
          </View>
        )}
      </View>

      <View style={styles.actionButtons}>
        <TouchableOpacity
          style={styles.actionButtonSecondary}
          onPress={onScanAgain}
        >
          <Text style={styles.actionButtonSecondaryText}>Scan Again</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.actionButtonPrimary}
          onPress={handleSaveReport}   // 👈 generate + share PDF
        >
          <Text style={styles.actionButtonPrimaryText}>Save Report</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}
