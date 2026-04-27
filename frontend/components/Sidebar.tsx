"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

type IconProps = { className?: string };

function DashboardIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <rect x="3" y="3" width="7" height="9" rx="1.5" />
      <rect x="14" y="3" width="7" height="5" rx="1.5" />
      <rect x="14" y="12" width="7" height="9" rx="1.5" />
      <rect x="3" y="16" width="7" height="5" rx="1.5" />
    </svg>
  );
}

function UploadIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

function MapIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <polygon points="1 6 8 3 16 6 23 3 23 18 16 21 8 18 1 21 1 6" />
      <line x1="8" y1="3" x2="8" y2="18" />
      <line x1="16" y1="6" x2="16" y2="21" />
    </svg>
  );
}

function ShieldIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

function MenuIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <line x1="3" y1="6" x2="21" y2="6" />
      <line x1="3" y1="12" x2="21" y2="12" />
      <line x1="3" y1="18" x2="21" y2="18" />
    </svg>
  );
}

function CloseIcon({ className }: IconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

const LINKS = [
  { href: "/", label: "Dashboard", icon: DashboardIcon },
  { href: "/upload", label: "Upload", icon: UploadIcon },
  { href: "/map", label: "Map", icon: MapIcon },
  { href: "/precautions", label: "Precautions", icon: ShieldIcon },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(href + "/");

  return (
    <>
      {/* Mobile top bar */}
      <div className="lg:hidden sticky top-0 z-40 flex items-center justify-between h-14 px-4 border-b border-zinc-800 bg-zinc-950">
        <span className="font-semibold tracking-tight text-zinc-100">DAMAGESCOPE</span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-label={open ? "Close navigation" : "Open navigation"}
          className="p-2 rounded-md text-zinc-300 hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {open ? <CloseIcon className="w-5 h-5" /> : <MenuIcon className="w-5 h-5" />}
        </button>
      </div>

      {/* Mobile drawer overlay */}
      {open && (
        <div
          className="lg:hidden fixed inset-0 z-30 bg-black/60"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={[
          "fixed inset-y-0 left-0 z-40 w-60 border-r border-zinc-800 bg-zinc-950 flex flex-col",
          "transition-transform duration-200 ease-out",
          open ? "translate-x-0" : "-translate-x-full",
          "lg:translate-x-0",
        ].join(" ")}
      >
        <div className="h-14 flex items-center px-5 border-b border-zinc-800">
          <div className="flex flex-col leading-tight">
            <span className="text-sm font-semibold tracking-tight text-zinc-100">
              DAMAGESCOPE
            </span>
            <span className="text-[10px] uppercase tracking-wider text-zinc-500">
              Damage Assessment
            </span>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {LINKS.map(({ href, label, icon: Icon }) => {
            const active = isActive(href);
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setOpen(false)}
                className={[
                  "group relative flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  active
                    ? "bg-zinc-800 text-white"
                    : "text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900",
                ].join(" ")}
                aria-current={active ? "page" : undefined}
              >
                {active && (
                  <span
                    aria-hidden="true"
                    className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r bg-blue-500"
                  />
                )}
                <Icon className="w-4 h-4 shrink-0" />
                <span className="font-medium">{label}</span>
              </Link>
            );
          })}
        </nav>

        <div className="border-t border-zinc-800 px-4 py-3 space-y-1">
          <div className="flex items-center gap-2 text-xs text-zinc-400">
            <span
              aria-hidden="true"
              className="w-1.5 h-1.5 rounded-full bg-emerald-500"
            />
            <span>Model online</span>
          </div>
          <p className="text-[10px] uppercase tracking-wider text-zinc-500">
            Siamese ResNet18 · v0.1
          </p>
        </div>
      </aside>
    </>
  );
}
