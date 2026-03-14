/**
 * Brief toast notification for incoming task nudges.
 *
 * Auto-dismisses after a configurable duration.
 * Keyed by _nudgeKey in parent — each new nudge mounts a fresh instance.
 * Uses useEffect for auto-dismiss timer (external timer subscription pattern).
 */
import { useCallback, useEffect } from 'react';
import { INTENT_CONFIG } from '../utils/constants';

const AUTO_DISMISS_MS = 4000;

export function NudgeToast({ nudge, onDismiss }) {
  // Auto-dismiss timer — subscribes to external timeout, calls parent callback
  useEffect(() => {
    if (!nudge) return;
    const timer = setTimeout(() => onDismiss?.(), AUTO_DISMISS_MS);
    return () => clearTimeout(timer);
  }, [nudge, onDismiss]);

  const handleClose = useCallback(() => {
    onDismiss?.();
  }, [onDismiss]);

  if (!nudge) return null;

  const intent = INTENT_CONFIG[nudge.intent?.toUpperCase()] || INTENT_CONFIG.TASK;

  return (
    <div
      className="nudge-toast nudge-toast--visible"
      role="status"
      aria-live="polite"
    >
      <div className="nudge-toast__icon" style={{ color: intent.color }}>
        {intent.icon}
      </div>
      <div className="nudge-toast__body">
        <span className="nudge-toast__label">New task surfaced</span>
        <span className="nudge-toast__preview">
          {nudge.content?.length > 60
            ? nudge.content.slice(0, 60) + '…'
            : nudge.content}
        </span>
      </div>
      <button
        className="nudge-toast__close"
        onClick={handleClose}
        aria-label="Dismiss notification"
      >
        ✕
      </button>
    </div>
  );
}
