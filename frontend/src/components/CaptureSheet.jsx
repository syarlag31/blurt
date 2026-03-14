/**
 * CaptureSheet — Bottom sheet overlay for text input + voice capture.
 *
 * Features:
 * - Slide-up animation with spring-like cubic-bezier
 * - Backdrop overlay with fade-in
 * - Dismiss via: swipe down, tap backdrop, or explicit close
 * - Text mode: auto-expanding textarea + send button
 * - Voice mode: large mic button with recording state + waveform
 * - Drag handle with gesture feedback via @use-gesture/react
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useDrag } from '@use-gesture/react';
import { Mic, Send, Type, Square } from 'lucide-react';
import './CaptureSheet.css';

const DISMISS_THRESHOLD = 100; // px to swipe before auto-dismiss

/**
 * @param {Object} props
 * @param {boolean} props.open - Whether sheet is visible
 * @param {() => void} props.onClose - Called when sheet should close
 * @param {(text: string) => void} props.onSendText - Submit text blurt
 * @param {() => void} props.onStartRecording - Begin voice capture
 * @param {() => void} props.onStopRecording - End voice capture
 * @param {boolean} props.recording - Whether voice is currently recording
 * @param {boolean} [props.disabled] - Disable inputs (e.g., no WS connection)
 */
export function CaptureSheet({
  open,
  onClose,
  onSendText,
  onStartRecording,
  onStopRecording,
  recording = false,
  disabled = false,
}) {
  const [mode, setMode] = useState('text'); // 'text' | 'voice'
  const [text, setText] = useState('');
  const [dragY, setDragY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef(null);
  const sheetRef = useRef(null);

  // Focus textarea when opening in text mode
  useEffect(() => {
    if (open && mode === 'text') {
      // Small delay to allow animation to start
      const timer = setTimeout(() => {
        textareaRef.current?.focus();
      }, 300);
      return () => clearTimeout(timer);
    }
  }, [open, mode]);

  // Reset state when closing
  useEffect(() => {
    if (!open) {
      setDragY(0);
      setIsDragging(false);
    }
  }, [open]);

  // ── Text submission ─────────────────────────────────────────────
  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSendText(trimmed);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    onClose();
  }, [text, disabled, onSendText, onClose]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleTextareaInput = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }
  }, []);

  // ── Voice toggle ────────────────────────────────────────────────
  const handleVoiceToggle = useCallback(() => {
    if (disabled) return;
    if (recording) {
      onStopRecording();
    } else {
      onStartRecording();
    }
  }, [disabled, recording, onStartRecording, onStopRecording]);

  // ── Dismiss handler ─────────────────────────────────────────────
  const handleDismiss = useCallback(() => {
    if (recording) {
      onStopRecording();
    }
    onClose();
  }, [recording, onStopRecording, onClose]);

  // ── Drag gesture for swipe-to-dismiss ───────────────────────────
  const bindDrag = useDrag(
    ({ movement: [, my], velocity: [, vy], direction: [, dy], cancel, active }) => {
      // Only allow downward drag
      const clampedY = Math.max(0, my);

      if (active) {
        setIsDragging(true);
        setDragY(clampedY);
      } else {
        setIsDragging(false);
        // Dismiss if dragged past threshold or fast flick downward
        if (clampedY > DISMISS_THRESHOLD || (vy > 0.5 && dy > 0)) {
          setDragY(0);
          handleDismiss();
        } else {
          // Snap back
          setDragY(0);
        }
      }
    },
    {
      axis: 'y',
      filterTaps: true,
      from: () => [0, 0],
    },
  );

  // ── Render nothing if not open (but allow exit animation) ───────
  // We use CSS transitions, so we need to keep the DOM present briefly
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    if (open) {
      setMounted(true);
    } else {
      // Delay unmount to allow exit animation
      const timer = setTimeout(() => setMounted(false), 350);
      return () => clearTimeout(timer);
    }
  }, [open]);

  if (!mounted) return null;

  const sheetStyle = isDragging
    ? { transform: `translateY(${dragY}px)` }
    : undefined;

  return (
    <>
      {/* Backdrop */}
      <div
        className={`capture-sheet-backdrop ${open ? 'capture-sheet-backdrop--visible' : ''}`}
        onClick={handleDismiss}
        aria-hidden="true"
      />

      {/* Sheet */}
      <div
        ref={sheetRef}
        className={[
          'capture-sheet',
          open ? 'capture-sheet--open' : '',
          isDragging ? 'capture-sheet--dragging' : '',
        ].join(' ')}
        style={sheetStyle}
        role="dialog"
        aria-modal="true"
        aria-label="Capture input"
      >
        {/* Drag handle */}
        <div className="capture-sheet__handle" {...bindDrag()}>
          <div className="capture-sheet__handle-bar" />
        </div>

        <div className="capture-sheet__body">
          {/* Mode toggle */}
          <div className="capture-sheet__mode-row">
            <button
              className={`capture-sheet__mode-btn ${mode === 'text' ? 'capture-sheet__mode-btn--active' : ''}`}
              onClick={() => setMode('text')}
              aria-pressed={mode === 'text'}
            >
              <Type className="capture-sheet__mode-icon" />
              Text
            </button>
            <button
              className={`capture-sheet__mode-btn ${mode === 'voice' ? 'capture-sheet__mode-btn--active' : ''}`}
              onClick={() => setMode('voice')}
              aria-pressed={mode === 'voice'}
            >
              <Mic className="capture-sheet__mode-icon" />
              Voice
            </button>
          </div>

          {/* Text input mode */}
          {mode === 'text' && (
            <div className="capture-sheet__input-row">
              <textarea
                ref={textareaRef}
                className="capture-sheet__textarea"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                onInput={handleTextareaInput}
                placeholder="Just blurt it out..."
                rows={1}
                disabled={disabled}
                aria-label="Type a blurt"
              />
              <button
                className="capture-sheet__send-btn"
                onClick={handleSubmit}
                disabled={disabled || !text.trim()}
                aria-label="Send blurt"
              >
                <Send className="capture-sheet__send-icon" />
              </button>
            </div>
          )}

          {/* Voice capture mode */}
          {mode === 'voice' && (
            <div className="capture-sheet__voice-area">
              {recording && (
                <div className="capture-sheet__waveform">
                  <div className="capture-sheet__waveform-bar" />
                  <div className="capture-sheet__waveform-bar" />
                  <div className="capture-sheet__waveform-bar" />
                  <div className="capture-sheet__waveform-bar" />
                  <div className="capture-sheet__waveform-bar" />
                </div>
              )}

              <button
                className={`capture-sheet__voice-btn ${recording ? 'capture-sheet__voice-btn--recording' : ''}`}
                onClick={handleVoiceToggle}
                disabled={disabled}
                aria-label={recording ? 'Stop recording' : 'Start recording'}
              >
                {recording ? (
                  <Square className="capture-sheet__voice-icon" />
                ) : (
                  <Mic className="capture-sheet__voice-icon" />
                )}
              </button>

              <span
                className={`capture-sheet__voice-label ${recording ? 'capture-sheet__voice-label--recording' : ''}`}
              >
                {recording ? 'Recording... tap to stop' : 'Tap to start recording'}
              </span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
