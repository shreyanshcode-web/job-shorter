import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Talent Intelligence",
  description: "AI candidate matching and ranking console",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
