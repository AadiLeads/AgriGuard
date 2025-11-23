declare module "react-native-rsa-native" {
  const RSA: {
    generateKeys: (keySize: number) => Promise<{ private: string; public: string }>;
    sign: (message: string, privateKey: string, algorithm: string) => Promise<string>;
    verify: (signature: string, message: string, publicKey: string, algorithm: string) => Promise<boolean>;
    encrypt: (message: string, publicKey: string) => Promise<string>;
    decrypt: (cipher: string, privateKey: string) => Promise<string>;
  };

  export default RSA;
}
