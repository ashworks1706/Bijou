import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const codecPro = localFont({
  src: "../public/codec-pro.regular.ttf",
  variable: "--font-codec-pro",
  weight: "400",
});

export const metadata: Metadata = {
  title: "Bijou - Tiny language models for tiny devices",
  description: "On-device AI optimized for headphones, wearables, AR glasses, and low-power hardware.",
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
