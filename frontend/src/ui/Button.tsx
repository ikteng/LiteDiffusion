import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cx } from "../lib/cx";

type Variant = "primary" | "solid" | "outline" | "ghost";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-accent-ink font-semibold hover:bg-accent-soft active:bg-accent disabled:bg-raised disabled:text-faint",
  solid: "bg-raised text-ink hover:bg-line disabled:text-faint",
  outline: "border border-line bg-surface text-ink hover:border-line-strong hover:bg-raised disabled:text-faint",
  ghost: "text-muted hover:bg-raised hover:text-ink disabled:text-faint",
};

const SIZES: Record<Size, string> = {
  sm: "h-8 gap-1.5 rounded-lg px-2.5 text-[13px]",
  md: "h-9 gap-2 rounded-lg px-3 text-sm",
  lg: "h-11 gap-2 rounded-xl px-4 text-[15px]",
};

const ICON_SIZES: Record<Size, string> = {
  sm: "w-8 px-0",
  md: "w-9 px-0",
  lg: "w-11 px-0",
};

type Props = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: Size;
  iconOnly?: boolean;
  children?: ReactNode;
};

export function Button({ variant = "solid", size = "md", iconOnly, className, ...rest }: Props) {
  return (
    <button
      type="button"
      {...rest}
      className={cx(
        "inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap transition-colors duration-100",
        "disabled:cursor-not-allowed disabled:opacity-70",
        VARIANTS[variant],
        SIZES[size],
        iconOnly && ICON_SIZES[size],
        className,
      )}
    />
  );
}
