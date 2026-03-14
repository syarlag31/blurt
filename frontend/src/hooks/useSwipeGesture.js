/**
 * useSwipeGesture — Reusable swipe gesture hook with momentum-based
 * transitions and snap-back spring physics.
 *
 * Built on @use-gesture/react's useDrag. Provides:
 * - Momentum-based dismiss: velocity amplifies displacement
 * - Rubber-band resistance beyond threshold
 * - Spring snap-back with overshoot (cubic-bezier)
 * - Configurable directions (horizontal, vertical, or both)
 * - 150-300ms animation durations per spec
 *
 * @param {Object}   options
 * @param {number}   [options.threshold=80]      - px to trigger action
 * @param {number}   [options.velocityThreshold=0.5] - velocity trigger
 * @param {string[]} [options.directions=['left','right']] - allowed directions
 * @param {number}   [options.dismissDistance=1.5] - factor of card size for exit
 * @param {number}   [options.resistance=0.35]   - rubber-band factor past threshold
 * @param {Function} [options.onSwipe]           - (direction) => void
 * @param {boolean}  [options.disabled=false]    - disable gesture
 *
 * @returns {{ bind, offset, direction, progress, isDismissing, isThresholdMet, springTransition, reset }}
 */
import { useState, useRef, useCallback } from 'react';
import { useDrag } from '@use-gesture/react';

/**
 * Spring-back transition: 200ms with slight overshoot for organic feel.
 * Cubic-bezier(0.175, 0.885, 0.32, 1.275) — "back-out" easing.
 */
const SPRING_BACK = 'transform 200ms cubic-bezier(0.175, 0.885, 0.32, 1.275)';

/**
 * Dismiss transition: 250ms fast ease-out with momentum feel.
 */
const DISMISS_TRANSITION =
  'transform 250ms cubic-bezier(0.32, 0.72, 0, 1), opacity 200ms ease-out';

/**
 * Resolve the dominant direction from a displacement vector.
 * Returns 'left', 'right', 'up', 'down', or null.
 */
function resolveDir(mx, my, allowedDirs) {
  const ax = Math.abs(mx);
  const ay = Math.abs(my);

  // Upward dominates when vertical is greater and my < 0
  if (my < 0 && ay > ax * 1.1 && ay > 15 && allowedDirs.includes('up')) return 'up';
  // Downward
  if (my > 0 && ay > ax * 1.1 && ay > 15 && allowedDirs.includes('down')) return 'down';
  // Horizontal
  if (ax > ay * 0.6 && ax > 12) {
    if (mx > 0 && allowedDirs.includes('right')) return 'right';
    if (mx < 0 && allowedDirs.includes('left')) return 'left';
  }
  return null;
}

function isHorizontal(dir) {
  return dir === 'left' || dir === 'right';
}

export function useSwipeGesture({
  threshold = 80,
  velocityThreshold = 0.5,
  directions = ['left', 'right'],
  dismissDistance = 1.5,
  resistance = 0.35,
  onSwipe,
  disabled = false,
} = {}) {
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDismissing, setIsDismissing] = useState(false);
  const [currentDir, setCurrentDir] = useState(null);
  const containerRef = useRef(null);
  const dismissTimerRef = useRef(null);

  // Reset to initial state
  const reset = useCallback(() => {
    setOffset({ x: 0, y: 0 });
    setIsDismissing(false);
    setCurrentDir(null);
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
  }, []);

  // Trigger dismiss animation + callback
  const triggerDismiss = useCallback(
    (direction) => {
      const el = containerRef.current;
      const w = el?.offsetWidth || 350;
      const h = el?.offsetHeight || 120;

      let targetX = 0;
      let targetY = 0;
      if (direction === 'right') targetX = w * dismissDistance;
      if (direction === 'left') targetX = -w * dismissDistance;
      if (direction === 'up') targetY = -(h * dismissDistance + 80);
      if (direction === 'down') targetY = h * dismissDistance + 80;

      setIsDismissing(true);
      setOffset({ x: targetX, y: targetY });

      // Fire callback after exit animation (250ms transition)
      dismissTimerRef.current = setTimeout(() => {
        onSwipe?.(direction);
      }, 260);
    },
    [dismissDistance, onSwipe],
  );

  // Drag handler
  const bind = useDrag(
    ({ down, movement: [mx, my], velocity: [vx, vy], cancel, memo }) => {
      if (disabled || isDismissing) {
        cancel();
        return memo;
      }

      if (down) {
        const dir = resolveDir(mx, my, directions);
        setCurrentDir(dir);

        // Apply rubber-band resistance beyond threshold
        const rubberBand = (val) => {
          const abs = Math.abs(val);
          if (abs > threshold) {
            return Math.sign(val) * (threshold + (abs - threshold) * resistance);
          }
          return val;
        };

        if (dir && isHorizontal(dir)) {
          setOffset({ x: rubberBand(mx), y: my * 0.12 });
        } else if (dir === 'up' || dir === 'down') {
          setOffset({ x: mx * 0.12, y: rubberBand(my) });
        } else {
          setOffset({ x: mx, y: my });
        }
      } else {
        // Released — check for dismiss
        const dir = resolveDir(mx, my, directions);

        if (dir) {
          const distance = isHorizontal(dir) ? Math.abs(mx) : Math.abs(my);
          const speed = isHorizontal(dir) ? Math.abs(vx) : Math.abs(vy);

          // Momentum: fast flick triggers at lower distance
          const momentumThreshold = speed > velocityThreshold
            ? threshold * 0.35
            : threshold;

          if (distance > momentumThreshold || (speed > velocityThreshold && distance > 25)) {
            triggerDismiss(dir);
            return memo;
          }
        }

        // Spring snap-back
        setOffset({ x: 0, y: 0 });
        setCurrentDir(null);
      }

      return memo;
    },
    {
      filterTaps: true,
      from: () => [0, 0],
      rubberband: false,
    },
  );

  // Computed values for consumers
  const direction = currentDir;
  const distance = direction
    ? isHorizontal(direction) ? Math.abs(offset.x) : Math.abs(offset.y)
    : Math.max(Math.abs(offset.x), Math.abs(offset.y));
  const progress = Math.min(distance / threshold, 1);
  const isThresholdMet = distance >= threshold;

  // Slight rotation on horizontal swipe for physical feedback
  const rotation = direction && isHorizontal(direction)
    ? (offset.x / 400) * 3
    : 0;

  // Build the inline transform + transition
  const transform = `translate3d(${offset.x}px, ${offset.y}px, 0) rotate(${rotation}deg)`;
  const transition = isDismissing
    ? DISMISS_TRANSITION
    : (offset.x === 0 && offset.y === 0)
      ? SPRING_BACK
      : 'none';

  const springStyle = {
    transform,
    transition,
    opacity: isDismissing ? 0 : 1,
    willChange: 'transform',
    touchAction: 'pan-y',
  };

  return {
    bind,
    ref: containerRef,
    offset,
    direction,
    progress,
    isDismissing,
    isThresholdMet,
    rotation,
    springStyle,
    reset,
    // Export timing constants for CSS coordination
    SPRING_BACK,
    DISMISS_TRANSITION,
  };
}
