"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Dashboard" },
  { href: "/upload", label: "Upload" },
  { href: "/map", label: "Map" },
  { href: "/precautions", label: "Precautions" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <header className="border-b border-zinc-800 bg-zinc-900">
      <div className="mx-auto max-w-6xl px-4 flex items-center gap-8 h-14">
        <span className="font-bold text-lg tracking-tight text-white">
          DAMAGESCOPE
        </span>
        <nav className="flex gap-1">
          {LINKS.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                pathname === href
                  ? "bg-zinc-700 text-white"
                  : "text-zinc-400 hover:text-white hover:bg-zinc-800"
              }`}
            >
              {label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
