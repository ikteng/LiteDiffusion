import type { ComponentProps, ReactNode } from "react";
import { Button as Base } from "@base-ui/react/button";
import { motion } from "framer-motion";
import { cx } from "../lib/cx";
import { POP } from "../lib/motion";

type Variant = "primary" | "solid" | "outline" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

/**
 * Every variant carries a 1px inset highlight along its top edge. That single line is what makes a flat dark rectangle
 * read as a raised, pressable surface, and it is why the buttons are described by shadows rather than by borders.
 */
const VARIANTS: Record<Variant, string> = {
  primary: cx(
    "bg-linear-to-b from-accent-soft to-accent text-accent-ink font-semibold",
    "shadow-[inset_0_1px_0_rgb(255_255_255/0.28),0_1px_2px_rgb(0_0_0/0.45)]",
    "hover:brightness-[1.06] data-pressed:brightness-[0.96]",
    "disabled:bg-none disabled:bg-raised disabled:text-faint disabled:shadow-none",
  ),
  solid: cx(
    "bg-raised text-ink",
    "shadow-[inset_0_1px_0_rgb(255_255_255/0.06),0_1px_2px_rgb(0_0_0/0.3)]",
    "hover:bg-line data-pressed:bg-raised",
    "disabled:text-faint disabled:shadow-none",
  ),
  outline: cx(
    "bg-surface text-ink ring-1 ring-inset ring-line",
    "hover:bg-raised hover:ring-line-strong data-pressed:bg-surface",
    "disabled:text-faint",
  ),
  ghost: cx("text-muted hover:bg-raised hover:text-ink data-pressed:bg-line", "disabled:text-faint"),
  danger: cx(
    "bg-bad/12 text-bad ring-1 ring-inset ring-bad/35",
    "hover:bg-bad/20 data-pressed:bg-bad/12",
    "disabled:text-faint disabled:ring-line",
  ),
};

/** Heights are on a 4px grid and the radius grows with them, so a row of mixed sizes still lines up. */
const SIZES: Record<Size, string> = {
  sm: "h-8 gap-1.5 rounded-lg px-2.5 text-[12.5px]",
  md: "h-9 gap-2 rounded-lg px-3 text-[13.5px]",
  lg: "h-10 gap-2 rounded-xl px-4 text-[14px]",
};

const ICON_SIZES: Record<Size, string> = { sm: "w-8 px-0", md: "w-9 px-0", lg: "w-10 px-0" };

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
 * Icons are sized by the button rather than by the caller (`[&_svg]:size-4`), which is the only way a toolbar of
 * buttons written by different hands ends up with icons that are actually the same size.
 *
 * The press is a spring rather than a CSS transition: a button that shrinks linearly and snaps back feels mechanical,
 * and the tiny overshoot on release is most of what makes a control feel physical. Base UI supplies `data-pressed`,
 * which keeps a keyboard-activated press looking the same as a pointer one.
 */
export function Button({ variant = "solid", size = "md", iconOnly, className, disabled, ...rest }: Props) {
  return (
    <Base
      type="button"
      disabled={disabled}
      {...rest}
      className={cx(
        "inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap",
        "transition-[background-color,color,filter,box-shadow] duration-100",
        "[&_svg]:size-4 [&_svg]:shrink-0",
        "disabled:cursor-not-allowed disabled:opacity-80",
        VARIANTS[variant],
        SIZES[size],
        iconOnly && ICON_SIZES[size],
        className,
      )}
      render={<motion.button whileTap={disabled ? undefined : { scale: 0.97 }} transition={POP} />}
    />
  );
}
