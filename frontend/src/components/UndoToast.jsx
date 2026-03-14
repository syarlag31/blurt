/**
 * UndoToast — 5-second undo notification with countdown timer.
 *
 * Shows a toast with:
 *  - Circular countdown ring + seconds remaining
 *  - Action message
 *  - "Undo" button
 *  - Close (X) button
 *  - Bottom progress bar
 *
 * Auto-dismisses after 5 seconds. Calls onUndo if user taps Undo,
 * or onDismiss when the timer expires or close is tapped.
 *
 * Props:
 *   message   — string (e.g. "Task archived")
 *   onUndo    — callback when undo is tapped
 *   onDismiss — callback when toast is dismissed (timer or close)
 *   duration  — ms, defaults to 5000
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import './UndoToast.css';

const DEFAULT_DURATION = 5000;
const CIRCUMFERENCE = 2 * Math.PI * 12; // r=12, ~75.4
const EXIT_ANIMATION_MS = 250;

export function UndoToast({ message, onUndo, onDismiss, duration = DEFAULT_DURATION }) {
  const [secondsLeft, setSecondsLeft] = useState(Math.ceil(duration / 1000));
  const [isDismissing, setIsDismissing] = useState(false);
  const [animateProgress, setAnimateProgress] = useState(false);
  const timerRef = useRef(null);
  const intervalRef = useRef(null);
  const startTimeRef = useRef(Date.now());
  const dismissedRef = useRef(false);

  const dismiss = useCallback(() => {
    if (dismissedRef.current) return;
    dismissedRef.current = true;
    clearTimeout(timerRef.current);
    clearInterval(intervalRef.current);
    setIsDismissing(true);
    setTimeout(() => onDismiss?.(), EXIT_ANIMATION_MS);
  }, [onDismiss]);

  const handleUndo = useCallback(() => {
    if (dismissedRef.current) return;
    dismissedRef.current = true;
    clearTimeout(timerRef.current);
    clearInterval(intervalRef.current);
    setIsDismissing(true);
    setTimeout(() => onUndo?.(), EXIT_ANIMATION_MS);
  }, [onUndo]);

  // Start the auto-dismiss timer
  useEffect(() => {
    startTimeRef.current = Date.now();

    // Kick off CSS transitions on next frame
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setAnimateProgress(true);
      });
    });

    // Countdown seconds display
    intervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTimeRef.current;
      const remaining = Math.ceil((duration - elapsed) / 1000);
      setSecondsLeft(Math.max(0, remaining));
    }, 250);

    // Auto-dismiss
    timerRef.current = setTimeout(() => {
      dismiss();
    }, duration);

    return () => {
      clearTimeout(timerRef.current);
      clearInterval(intervalRef.current);
    };
  }, [duration, dismiss]);

  const durationSec = duration / 1000;
  const ringOffset = animateProgress ? CIRCUMFERENCE : 0;
  const scaleX = animateProgress ? 0 : 1;

  return (
    <div
      className={`undo-toast${isDismissing ? ' undo-toast--dismissing' : ''}`}
      role="alert"
      aria-live="assertive"
    >
      {/* Countdown ring */}
      <div className="undo-toast__countdown">
        <svg className="undo-toast__countdown-ring" viewBox="0 0 28 28">
          <circle
            className="undo-toast__countdown-track"
            cx="14"
            cy="14"
            r="12"
          />
          <circle
            className={`undo-toast__countdown-progress${animateProgress ? ' undo-toast__countdown-progress--animate' : ''}`}
            cx="14"
            cy="14"
            r="12"
            style={{
              strokeDashoffset: ringOffset,
              transitionDuration: `${durationSec}s`,
            }}
          />
        </svg>
        <span className="undo-toast__countdown-text" aria-hidden="true">
          {secondsLeft}
        </span>
      </div>

      {/* Message */}
      <div className="undo-toast__body">
        <span className="undo-toast__message">{message}</span>
      </div>

      {/* Undo action */}
      <button
        className="undo-toast__action"
        onClick={handleUndo}
        aria-label={`Undo: ${message}`}
      >
        Undo
      </button>

      {/* Close */}
      <button
        className="undo-toast__close"
        onClick={dismiss}
        aria-label="Dismiss"
      >
        <svg
          className="undo-toast__close-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <line x1="18" y1="6" x2="6" y2="18" />
          <line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>

      {/* Bottom progress bar */}
      <div className="undo-toast__progress">
        <div
          className={`undo-toast__progress-bar${animateProgress ? ' undo-toast__progress-bar--animate' : ''}`}
          style={{
            transform: `scaleX(${scaleX})`,
            transitionDuration: `${durationSec}s`,
          }}
        />
      </div>
    </div>
  );
}
