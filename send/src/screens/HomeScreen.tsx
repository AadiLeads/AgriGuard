import React from "react";
import { View, Text, TouchableOpacity } from "react-native";
import {
  Leaf,
  ChevronRight,
  CheckCircle2,
  AlertTriangle,
  Droplets,
  Sun,
} from "lucide-react-native";
import { styles } from "../styles/agriGuardStyles";

type Props = { onScanClick: () => void };

export default function HomeScreen({ onScanClick }: Props) {
  return (
    <View style={styles.screen}>
      <View style={styles.welcomeSection}>
        <Text style={styles.welcomeTitle}>
          Hello,{"\n"}
          <Text style={styles.welcomeTitleGreen}>Green Thumb! 🌿</Text>
        </Text>
        <Text style={styles.welcomeSubtitle}>
          Let's check your plants' health today.
        </Text>
      </View>

      <TouchableOpacity style={styles.ctaCard} onPress={onScanClick}>
        <View style={styles.ctaContent}>
          <Text style={styles.ctaTitle}>Scan Your Plant</Text>
          <Text style={styles.ctaSubtitle}>
            Detect diseases early with AI precision.
          </Text>
          <View style={styles.ctaButton}>
            <Text style={styles.ctaButtonText}>Start Scan</Text>
            <ChevronRight size={16} color="#fff" />
          </View>
        </View>
        <Leaf
          size={100}
          color="rgba(255,255,255,0.2)"
          style={styles.ctaLeaf}
        />
      </TouchableOpacity>

      <View style={styles.statsRow}>
        <View style={styles.statCard}>
          <View style={[styles.statIcon, { backgroundColor: "#D1FAE5" }]}>
            <CheckCircle2 size={20} color="#22C55E" />
          </View>
          <Text style={styles.statNumber}>12</Text>
          <Text style={styles.statLabel}>Healthy Plants</Text>
        </View>
        <View style={styles.statCard}>
          <View style={[styles.statIcon, { backgroundColor: "#FED7AA" }]}>
            <AlertTriangle size={20} color="#F97316" />
          </View>
          <Text style={styles.statNumber}>3</Text>
          <Text style={styles.statLabel}>Needs Attention</Text>
        </View>
      </View>

      <View style={styles.tipsSection}>
        <View style={styles.tipsSectionHeader}>
          <Text style={styles.tipsTitle}>Daily Tips</Text>
          <TouchableOpacity>
            <Text style={styles.tipsViewAll}>View All</Text>
          </TouchableOpacity>
        </View>
        <TipCard
          icon={<Droplets size={18} color="#3B82F6" />}
          title="Watering Schedule"
          desc="Most plants prefer early morning watering."
        />
        <TipCard
          icon={<Sun size={18} color="#F59E0B" />}
          title="Sunlight Check"
          desc="Ensure 6 hours of indirect light."
        />
      </View>
    </View>
  );
}

function TipCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <View style={styles.tipCard}>
      <View style={styles.tipIcon}>{icon}</View>
      <View style={styles.tipContent}>
        <Text style={styles.tipTitle}>{title}</Text>
        <Text style={styles.tipDesc}>{desc}</Text>
      </View>
    </View>
  );
}
