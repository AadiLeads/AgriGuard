import React from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import {
  X,
  User,
  History,
  Settings,
  AlertTriangle,
  LogOut,
  Leaf,
  ChevronRight,
} from "lucide-react-native";
import { styles } from "../styles/agriGuardStyles";

type Props = {
  onBack: () => void;
  onLogout: () => void;
};

export default function AboutScreen({ onBack, onLogout }: Props) {
  return (
    <View style={styles.screen}>
      <View style={styles.aboutHeader}>
        <TouchableOpacity onPress={onBack} style={styles.backButton}>
          <X size={20} color="#000" />
        </TouchableOpacity>
        <Text style={styles.aboutTitle}>Profile</Text>
        <View style={{ width: 40 }} />
      </View>

      <View style={styles.profileCard}>
        <View style={styles.profileAvatar}>
          <Text style={styles.profileEmoji}>👨‍🌾</Text>
        </View>
        <View>
          <Text style={styles.profileName}>Gardener Joe</Text>
          <Text style={styles.profileMembership}>Free Member</Text>
        </View>
      </View>

      <Text style={styles.sectionTitle}>SETTINGS</Text>
      <MenuItem
        icon={<User size={20} color="#000" />}
        label="Account Details"
        onPress={() => Alert.alert("Account Details", "Coming soon!")}
      />
      <MenuItem
        icon={<History size={20} color="#000" />}
        label="Scan History"
        onPress={() => Alert.alert("Scan History", "Coming soon!")}
      />
      <MenuItem
        icon={<Settings size={20} color="#000" />}
        label="Preferences"
        onPress={() => Alert.alert("Preferences", "Coming soon!")}
      />

      <Text style={styles.sectionTitle}>SUPPORT</Text>
      <MenuItem
        icon={<AlertTriangle size={20} color="#000" />}
        label="Report a Bug"
        onPress={() => Alert.alert("Report a Bug", "Coming soon!")}
      />
      <MenuItem
        icon={<LogOut size={20} color="#EF4444" />}
        label="Log Out"
        onPress={onLogout}
      />

      <View style={styles.footer}>
        <View style={styles.footerBrand}>
          <Leaf size={24} color="#22C55E" />
          <Text style={styles.footerBrandText}>AgriGuard</Text>
        </View>
        <Text style={styles.footerVersion}>
          Version 1.0.2 • Made with 💚 for Nature
        </Text>
      </View>
    </View>
  );
}

function MenuItem({
  icon,
  label,
  onPress,
}: {
  icon: React.ReactNode;
  label: string;
  onPress?: () => void;
}) {
  return (
    <TouchableOpacity style={styles.menuItem} onPress={onPress}>
      <View style={styles.menuItemLeft}>
        {icon}
        <Text style={styles.menuItemLabel}>{label}</Text>
      </View>
      <ChevronRight size={16} color="#999" />
    </TouchableOpacity>
  );
}
