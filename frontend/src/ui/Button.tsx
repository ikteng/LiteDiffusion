import type { ComponentProps, ReactNode } from "react";
import { Button as Base } from "@base-ui/react/button";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

type Variant = "primary" | "solid" | "outline" | "ghost";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-accent text-accent-ink font-semibold hover:bg-accent-soft data-pressed:bg-accent disabled:bg-raised disabled:text-faint",
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

// Base UI allows `className` to be a function of the component's state; every call site here passes a plain string,
// and narrowing it keeps `cx` usable.
type Props = Omit<ComponentProps<typeof Base>, "className" | "render"> & {
  className?: string;
  variant?: Variant;
  size?: Size;
  iconOnly?: boolean;
  children?: ReactNode;
};

/**
 * Every button in the app.
 *
 * The press is a spring rather than a CSS transition because a button that shrinks linearly and snaps back feels
 * mechanical; the tiny overshoot on release is most of what makes a control feel physical. Base UI supplies the
 * `data-pressed` state, which is what keeps the keyboard-activated press looking the same as the pointer one.
 */
export function Button({ variant = "solid", size = "md", iconOnly, className, disabled, ...rest }: Props) {
  return (
    <Base
      type="button"
      disabled={disabled}
      {...rest}
      className={cx(
        "inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap transition-colors duration-100",
        "disabled:cursor-not-allowed disabled:opacity-70",
        VARIANTS[variant],
        SIZES[size],
        iconOnly && ICON_SIZES[size],
        className,
      )}
      render={<motion.button whileTap={disabled ? undefined : { scale: 0.97 }} transition={POP} />}
    />
  );
}
