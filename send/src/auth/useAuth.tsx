// src/auth/useAuth.tsx
import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
} from "react";
import * as SecureStore from "expo-secure-store";

const API_BASE_URL = "https://dwb3r7h0-8000.inc1.devtunnels.ms";

type AuthContextValue = {
  token: string | null;
  csrf: string | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [csrf, setCsrf] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  async function login(username: string, password: string) {
    setIsLoading(true);
    try {
      const form = new FormData();
      form.append("username", username);
      form.append("password", password);

      const res = await fetch(`${API_BASE_URL}/auth/login`, {
        method: "POST",
        body: form,
      });

      const raw = await res.text();
      console.log("🔐 login status:", res.status);
      console.log("🔐 login raw body:", raw);

      if (!res.ok) {
        throw new Error(`Login failed: ${raw}`);
      }

      const data = JSON.parse(raw);
      console.log("🔐 parsed login data:", data);

      // 👇 adjust these keys to match your backend response
      const t: string | null =
        data.token ?? data.access_token ?? null;
      const c: string | null =
        data.csrf ?? data.csrf_token ?? null;

      if (!t || !c) {
        throw new Error(
          "Login response missing token/csrf fields. Check backend response keys."
        );
      }

      setToken(t);
      setCsrf(c);

      await SecureStore.setItemAsync("auth_token", t);
      await SecureStore.setItemAsync("auth_csrf", c);

      console.log("✅ stored token/csrf:", t, c);
    } finally {
      setIsLoading(false);
    }
  }

  async function register(username: string, password: string) {
    setIsLoading(true);
    try {
      const form = new FormData();
      form.append("username", username);
      form.append("password", password);

      const res = await fetch(`${API_BASE_URL}/auth/register`, {
        method: "POST",
        body: form,
      });

      const raw = await res.text();
      console.log("📝 register status:", res.status);
      console.log("📝 register raw body:", raw);

      if (!res.ok) {
        throw new Error(`Registration failed: ${raw}`);
      }
    } finally {
      setIsLoading(false);
    }
  }

  async function logout() {
    setToken(null);
    setCsrf(null);
    await SecureStore.deleteItemAsync("auth_token");
    await SecureStore.deleteItemAsync("auth_csrf");
    console.log("🚪 Logged out");
  }

  const value: AuthContextValue = {
    token,
    csrf,
    isLoading,
    login,
    register,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used inside <AuthProvider>");
  }
  return ctx;
}
