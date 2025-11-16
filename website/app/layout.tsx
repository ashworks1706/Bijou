import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const codecPro = localFont({
  src: "../public/codec-pro.regular.ttf",
  variable: "--font-codec-pro",
  weight: "400",
});

export const metadata: Metadata = {
  title: "Bijou - Tiny Language Models for Tiny Devices",
  description: "On-device AI optimized for headphones, wearables, AR glasses, and low-power hardware. Zero cloud, zero latency, zero hallucination. Open-source framework for specialized LLMs.",
  keywords: ["on-device AI", "tiny language models", "wearables", "edge AI", "offline AI", "quantized models", "tool-calling", "function-calling", "small language models", "SLM"],
  authors: [{ name: "Bijou Team" }],
  openGraph: {
    title: "Bijou - Tiny Language Models for Tiny Devices",
    description: "On-device AI optimized for headphones, wearables, AR glasses, and low-power hardware. Zero cloud, zero latency, zero hallucination.",
    url: "https://bijou.ai",
    siteName: "Bijou",
    type: "website",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Bijou - Tiny Language Models for Tiny Devices",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Bijou - Tiny Language Models for Tiny Devices",
    description: "On-device AI optimized for headphones, wearables, AR glasses, and low-power hardware.",
    images: ["/og-image.png"],
  },
  robots: {
    index: true,
    follow: true,
  },
  metadataBase: new URL("https://bijou.ai"),
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${codecPro.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
