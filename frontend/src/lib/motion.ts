import type { Transition } from "framer-motion";

/**
 * Three transitions for the whole app, so motion reads as one system rather than a per-component opinion.
 *
 * Reduced motion is handled once, by the `<MotionConfig reducedMotion="user">` in `main.tsx` — these do not need to
 * check for it.
 */

/** Things that appear where you just clicked: popups, thumbs, the segmented highlight. Fast, barely any overshoot. */
export const POP: Transition = { type: "spring", stiffness: 560, damping: 38, mass: 0.8 };

/** Things that travel: a block changing place, the composer sliding up out of the hero. Slower, so it is followable. */
export const SETTLE: Transition = { type: "spring", stiffness: 320, damping: 34, mass: 0.9 };

/** Things that only change opacity. A spring on a cross-fade is wasted, so this is a plain ease-out. */
export const FADE: Transition = { duration: 0.16, ease: [0.22, 1, 0.36, 1] };
