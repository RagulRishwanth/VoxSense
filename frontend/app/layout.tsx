import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: "VoxSense — Interactive AI for Parkinson's Voice Biomarker Analysis",
  description: "Interactive, explainable AI workspace for analyzing pre-extracted Parkinson's vocal acoustic biomarkers and longitudinal UPDRS trajectories. Research and educational use only.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}

