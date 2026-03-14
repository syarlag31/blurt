/**
 * CaptureFab — Persistent floating action button for quick blurt capture.
 *
 * Features:
 * - Fixed position bottom-right, above the BottomTabBar
 * - Plus icon that rotates to X when sheet is open
 * - Subtle pulse ring for ambient attention
 * - Scale + glow micro-interactions on tap
 * - Bottom sheet with text input for quick capture
 * - Sends text.input via WebSocket for processing
 * - Premium dark theme with blue-tinted glow
 * - 56px touch target (exceeds 44px minimum)
 * - Reduced motion support
 */
import { useState, useRef, useCallback, useEffect } from 'react';
import { Plus, Send } from 'lucide-react';
import './CaptureFAB.css';

export function CaptureFAB({ onSubmit, visible = true }) {
  const [isOpen, setIsOpen] = useState(false);
  const [text, setText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const textareaRef = useRef(null);

  // Focus textarea when sheet opens
  useEffect(() => {
    if (isOpen && textareaRef.current) {
      // Small delay so the slide animation is underway
      const timer = setTimeout(() => {
        textareaRef.current?.focus();
      }, 350);
      return () => clearTimeout(timer);
    }
  }, [isOpen]);

  const handleToggle = useCallback(() => {
    setIsOpen((prev) => !prev);
    if (isOpen) {
      setText('');
    }
  }, [isOpen]);

  const handleBackdropClick = useCallback(() => {
    setIsOpen(false);
    setText('');
  }, []);

  const handleSubmit = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed || isSubmitting) return;

    setIsSubmitting(true);
    try {
      if (onSubmit) {
        await onSubmit(trimmed);
      }
      setText('');
      setIsOpen(false);
    } finally {
      setIsSubmitting(false);
    }
  }, [text, isSubmitting, onSubmit]);

  const handleKeyDown = useCallback(
    (e) => {
      // Submit on Enter (without Shift for newline)
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  // Close sheet if FAB becomes hidden
  useEffect(() => {
    if (!visible && isOpen) {
      setIsOpen(false);
      setText('');
    }
  }, [visible, isOpen]);

  if (!visible) return null;

  return (
    <>
      {/* Bottom sheet overlay */}
      <div
        className={`capture-overlay ${isOpen ? 'capture-overlay--visible' : ''}`}
        aria-hidden={!isOpen}
      >
        <div
          className="capture-overlay__backdrop"
          onClick={handleBackdropClick}
        />
        <div className="capture-overlay__sheet" role="dialog" aria-label="Quick capture">
          <div className="capture-overlay__handle" />
          <h2 className="capture-overlay__title">Quick Capture</h2>
          <textarea
            ref={textareaRef}
            className="capture-overlay__input"
            placeholder="What's on your mind?"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={3}
            disabled={isSubmitting}
          />
          <button
            className="capture-overlay__submit"
            onClick={handleSubmit}
            disabled={!text.trim() || isSubmitting}
            aria-label="Send blurt"
          >
            <Send size={18} />
            {isSubmitting ? 'Sending...' : 'Blurt it'}
          </button>
        </div>
      </div>

      {/* FAB button */}
      <button
        className={`capture-fab ${isOpen ? 'capture-fab--open' : ''}`}
        onClick={handleToggle}
        aria-label={isOpen ? 'Close capture' : 'Quick capture'}
        aria-expanded={isOpen}
      >
        {!isOpen && <span className="capture-fab__pulse" aria-hidden="true" />}
        <span className="capture-fab__icon">
          <Plus size={24} strokeWidth={2.5} />
        </span>
      </button>
    </>
  );
}
