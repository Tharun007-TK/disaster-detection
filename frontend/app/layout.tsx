import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "DAMAGESCOPE",
  description: "Disaster damage assessment using Siamese ResNet18",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} h-full`}>
      <body className="min-h-screen bg-zinc-950 text-zinc-100 antialiased">
        <Sidebar />
        <div className="lg:pl-60 min-h-screen flex flex-col">
          <main className="flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
