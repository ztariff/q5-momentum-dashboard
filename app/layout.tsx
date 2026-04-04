import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Q5 Momentum — Strategy Dashboard",
  description: "Live strategy dashboard for Q5 Momentum trading strategy",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased min-h-screen" style={{ backgroundColor: '#0a0e1a', color: '#e2e8f0' }}>
        {children}
      </body>
    </html>
  );
}
