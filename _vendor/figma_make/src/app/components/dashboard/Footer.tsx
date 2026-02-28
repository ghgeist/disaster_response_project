import React, { type ComponentType } from "react";
import { Globe } from "lucide-react";

type FooterIconProps = {
  size?: number;
  className?: string;
};

const GitHubIcon = ({ size = 18, className }: FooterIconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-hidden="true"
  >
    <path
      d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.013-1.703-2.782.604-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0 1 12 6.836a9.59 9.59 0 0 1 2.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.579.688.481C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"
      fill="currentColor"
    />
  </svg>
);

const LinkedInIcon = ({ size = 18, className }: FooterIconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="currentColor"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-hidden="true"
  >
    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
  </svg>
);

const SubstackIcon = ({ size = 18, className }: FooterIconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    aria-hidden="true"
  >
    <path
      d="M22.539 8.242H1.46V5.406h21.08v2.836zM1.46 10.812V24L12 18.11 22.54 24V10.812H1.46zM22.54 0H1.46v2.836h21.08V0z"
      fill="currentColor"
    />
  </svg>
);

const currentYear = new Date().getFullYear();

type FooterLink = {
  name: string;
  href: string;
  Icon: ComponentType<FooterIconProps>;
};

const socialLinks: FooterLink[] = [
  {
    name: "GitHub",
    href: "https://github.com/ghgeist/disaster_response_project",
    Icon: GitHubIcon,
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/grantgeist/",
    Icon: LinkedInIcon,
  },
  {
    name: "Substack",
    href: "https://thedonkeyaxiom.substack.com/",
    Icon: SubstackIcon,
  },
  {
    name: "Website",
    href: "https://grantgeist.com",
    Icon: Globe,
  },
];

export function Footer() {
  return (
    <footer role="contentinfo" className="border-t border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-3 px-4 py-3 text-xs text-slate-600 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>© {currentYear} Grant Geist</span>
        </div>

        <div className="flex items-center gap-2">
          {socialLinks.map(({ name, href, Icon }) => (
            <a
              key={name}
              href={href}
              target={href.startsWith("http") ? "_blank" : undefined}
              rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
              aria-label={name}
              className="group relative rounded-md border border-transparent p-1.5 text-slate-500 transition-colors hover:border-slate-200 hover:bg-slate-50 hover:text-slate-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            >
              <span className="pointer-events-none absolute -top-8 left-1/2 -translate-x-1/2 whitespace-nowrap rounded bg-slate-900 px-2 py-1 text-[10px] font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
                {name}
              </span>
              <Icon size={16} />
            </a>
          ))}
        </div>
      </div>
    </footer>
  );
}
