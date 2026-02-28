import { type ComponentType } from "react";
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
    href: "https://github.com/ghgeist",
    Icon: GitHubIcon,
  },
  {
    name: "Substack",
    href: "https://thedonkeyaxiom.substack.com/",
    Icon: SubstackIcon,
  },
  {
    name: "Website",
    href: "https://granthgeist.com",
    Icon: Globe,
  },
];

export function Footer() {
  return (
    <footer role="contentinfo" className="border-t border-slate-200 bg-white/95 backdrop-blur-sm">
      <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-3 px-4 py-3 text-xs text-slate-600 md:flex-row md:items-center md:justify-between">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <span>Data: EPA National Walkability Index</span>
          <span aria-hidden="true" className="text-slate-400">·</span>
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
