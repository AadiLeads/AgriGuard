// App.tsx
import React from "react";
import { AuthProvider } from "./src/auth/useAuth"; // adjust path if needed
import { AgriGuardApp } from "./src/AgriGuardApp";

export default function App() {
  return (
    <AuthProvider>
      <AgriGuardApp />
    </AuthProvider>
  );
}
