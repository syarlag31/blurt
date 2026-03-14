/**
 * Voice input button with dual interaction modes.
 *
 * Modes (via `mode` prop):
 *   "auto"   — (default) short tap toggles, long press records while held
 *   "toggle" — tap-to-toggle only (tap to start, tap to stop)
 *   "hold"   — press-and-hold only (hold to record, release to stop)
 *
 * Visual feedback: pulsing ring when recording, color change.
 */
import { useCallback, useRef, useState } from 'react';

const HOLD_THRESHOLD_MS = 300;

/** SVG microphone icon — idle state */
function MicIcon() {
  return (
    <svg
      className="voice-btn__svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="9" y="1" width="6" height="12" rx="3" />
      <path d="M19 10v1a7 7 0 0 1-14 0v-1" />
      <line x1="12" y1="18" x2="12" y2="23" />
      <line x1="8" y1="23" x2="16" y2="23" />
    </svg>
  );
}

/** SVG stop icon — recording state */
function StopIcon() {
  return (
    <svg
      className="voice-btn__svg"
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
    >
      <rect x="5" y="5" width="14" height="14" rx="2" />
    </svg>
  );
}

/**
 * @param {Object} props
 * @param {boolean} props.recording - Whether currently recording
 * @param {() => void} props.onToggle - Toggle recording on/off (used in toggle/auto modes)
 * @param {() => Promise<void>} props.onStart - Start recording (used in hold/auto modes)
 * @param {() => void} props.onStop - Stop recording (used in hold/auto modes)
 * @param {boolean} [props.disabled] - Disable the button
 * @param {"auto"|"toggle"|"hold"} [props.mode] - Interaction mode (default: "auto")
 */
export function VoiceButton({
  recording,
  onToggle,
  onStart,
  onStop,
  disabled,
  mode = 'auto',
}) {
  const [holdActive, setHoldActive] = useState(false);
  const pressTimerRef = useRef(null);
  const didHoldRef = useRef(false);

  // ── Hold-only mode handlers ──────────────────────────────────────
  const handleHoldDown = useCallback(
    async (e) => {
      if (disabled) return;
      e.preventDefault();
      if (!recording) {
        setHoldActive(true);
        await onStart();
      }
    },
    [disabled, recording, onStart],
  );

  const handleHoldUp = useCallback(
    (e) => {
      if (disabled) return;
      e.preventDefault();
      if (recording) {
        onStop();
      }
      setHoldActive(false);
    },
    [disabled, recording, onStop],
  );

  // ── Auto mode handlers (detect tap vs hold) ─────────────────────
  const handleAutoDown = useCallback(
    (e) => {
      if (disabled) return;
      e.preventDefault();
      didHoldRef.current = false;

      pressTimerRef.current = setTimeout(async () => {
        didHoldRef.current = true;
        setHoldActive(true);
        if (!recording) {
          await onStart();
        }
      }, HOLD_THRESHOLD_MS);
    },
    [disabled, recording, onStart],
  );

  const handleAutoUp = useCallback(
    (e) => {
      if (disabled) return;
      e.preventDefault();
      clearTimeout(pressTimerRef.current);

      if (didHoldRef.current) {
        // Was a press-and-hold — stop recording on release
        if (recording) {
          onStop();
        }
        setHoldActive(false);
      } else {
        // Was a tap — toggle recording
        onToggle();
      }
      didHoldRef.current = false;
    },
    [disabled, recording, onToggle, onStop],
  );

  // ── Shared cancel handler ────────────────────────────────────────
  const handleCancel = useCallback(() => {
    clearTimeout(pressTimerRef.current);
    if ((didHoldRef.current || holdActive) && recording) {
      onStop();
    }
    setHoldActive(false);
    didHoldRef.current = false;
  }, [recording, onStop, holdActive]);

  // ── Resolve event handlers based on mode ─────────────────────────
  let pointerDown;
  let pointerUp;

  if (mode === 'toggle') {
    // Toggle mode: simple click handler, no pointer tracking needed
    pointerDown = undefined;
    pointerUp = undefined;
  } else if (mode === 'hold') {
    pointerDown = handleHoldDown;
    pointerUp = handleHoldUp;
  } else {
    // auto (default)
    pointerDown = handleAutoDown;
    pointerUp = handleAutoUp;
  }

  const handleClick = mode === 'toggle' ? onToggle : undefined;

  // ── Title / label helpers ────────────────────────────────────────
  const label = recording ? 'Stop recording' : 'Start recording';
  const title =
    mode === 'toggle'
      ? recording
        ? 'Tap to stop'
        : 'Tap to record'
      : mode === 'hold'
        ? recording
          ? 'Release to stop'
          : 'Hold to record'
        : recording
          ? 'Release or tap to stop'
          : 'Tap to toggle, hold to record';

  return (
    <button
      className={`voice-btn ${recording ? 'voice-btn--recording' : ''} ${holdActive ? 'voice-btn--hold' : ''}`}
      onPointerDown={pointerDown}
      onPointerUp={pointerUp}
      onPointerCancel={mode !== 'toggle' ? handleCancel : undefined}
      onPointerLeave={mode !== 'toggle' ? handleCancel : undefined}
      onClick={handleClick}
      disabled={disabled}
      aria-label={label}
      title={title}
      data-mode={mode}
    >
      <span className="voice-btn__icon">
        {recording ? <StopIcon /> : <MicIcon />}
      </span>
      {recording && <span className="voice-btn__pulse" />}
    </button>
  );
}
