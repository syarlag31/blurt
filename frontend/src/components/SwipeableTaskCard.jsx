/**
 * SwipeableTaskCard — Wraps TaskCard with swipe gesture handlers.
 *
 * Swipe directions map to Thompson Sampling feedback actions:
 * - Right → engage (user wants to work on it now)
 * - Left  → defer  (push to later)
 * - Up    → done   (mark complete)
 *
 * Uses useSwipeGesture hook for momentum-based spring physics.
 * Shows directional color hints and action labels during swipe.
 * Calls POST /tasks/{task_id}/feedback with the mapped action.
 *
 * Animation timings: 200ms snap-back (spring overshoot), 250ms dismiss.
 */
import { useCallback } from 'react';
import { CheckCircle2, Clock, ArrowRight } from 'lucide-react';
import { TaskCard } from './TaskCard';
import { useSwipeGesture } from '../hooks/useSwipeGesture';
import { API_BASE, USER_ID } from '../utils/constants';
import './SwipeableTaskCard.css';

/* ── Action mapping: swipe direction → feedback action ─────────── */
const SWIPE_CONFIG = {
  right: { action: 'engage',   label: 'Engage', Icon: ArrowRight,    bgClass: 'engage',   color: 'var(--accent, #3b82f6)' },
  left:  { action: 'defer',    label: 'Defer',  Icon: Clock,         bgClass: 'defer',    color: 'var(--warning, #f59e0b)' },
  up:    { action: 'complete', label: 'Done',   Icon: CheckCircle2,  bgClass: 'done',     color: 'var(--success, #10b981)' },
};

/**
 * Submit Thompson Sampling feedback to the backend.
 * Best-effort: errors are logged but don't block the UI.
 */
async function submitFeedback(taskId, action) {
  try {
    const res = await fetch(
      `${API_BASE}/tasks/${encodeURIComponent(taskId)}/feedback`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: USER_ID, action }),
      },
    );
    if (!res.ok) {
      console.warn(`[SwipeableTaskCard] Feedback ${action} failed: ${res.status}`);
    }
  } catch (err) {
    console.warn('[SwipeableTaskCard] Feedback error:', err);
  }
}

export function SwipeableTaskCard({ task, onAction, onSwipeAction, weightsUsed }) {
  const taskId = task?.task_id || task?.id;

  const handleSwipe = useCallback(
    (direction) => {
      const config = SWIPE_CONFIG[direction];
      if (!config || !taskId) return;

      // Submit Thompson Sampling feedback
      submitFeedback(taskId, config.action);

      // Notify parent for optimistic removal + re-fetch
      onSwipeAction?.(config.action, taskId);
      onAction?.(config.action, taskId);
    },
    [taskId, onSwipeAction, onAction],
  );

  const {
    bind,
    ref,
    offset,
    direction: dir,
    progress,
    isDismissing,
    isThresholdMet,
    springStyle,
  } = useSwipeGesture({
    threshold: 90,
    velocityThreshold: 0.5,
    directions: ['left', 'right', 'up'],
    dismissDistance: 1.5,
    resistance: 0.35,
    onSwipe: handleSwipe,
  });

  // ── Computed display state ──────────────────────────────────────
  const activeConfig = dir ? SWIPE_CONFIG[dir] : null;

  return (
    <div
      className={`swipeable-task-card${isDismissing ? ' swipeable-task-card--dismissing' : ''}`}
      ref={ref}
    >
      {/* Action indicator background */}
      {activeConfig && progress > 0.05 && !isDismissing && (
        <div
          className={`swipeable-task-card__action-bg swipeable-task-card__action-bg--${activeConfig.bgClass}${
            isThresholdMet ? ' swipeable-task-card__action-bg--ready' : ''
          }`}
          style={{ opacity: progress * 0.85 + 0.15 }}
          aria-hidden="true"
        >
          <div className={`swipeable-task-card__action-label swipeable-task-card__action-label--${activeConfig.bgClass}`}>
            <activeConfig.Icon
              size={24}
              strokeWidth={2}
              style={{
                transform: `scale(${0.6 + progress * 0.4})`,
                transition: 'transform 150ms ease-out',
              }}
            />
            <span>{activeConfig.label}</span>
          </div>
        </div>
      )}

      {/* Swipeable card surface with momentum-based spring physics */}
      <div
        {...bind()}
        className="swipeable-task-card__surface"
        style={springStyle}
      >
        <TaskCard task={task} onAction={onAction} weightsUsed={weightsUsed} />
      </div>
    </div>
  );
}
