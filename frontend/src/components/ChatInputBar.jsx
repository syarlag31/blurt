/**
 * ChatInputBar — Premium text input bar with send and voice buttons.
 *
 * Features:
 *   - Auto-expanding textarea (up to 120px)
 *   - Lucide SVG icons for send and microphone
 *   - 44px+ touch targets for mobile UX
 *   - Animated send button that appears when text is entered
 *   - Voice button with recording pulse indicator
 *   - Premium dark theme styling with blue-tinted surfaces
 *   - Submit on Enter (shift+Enter for newline)
 */
import { useCallback, useRef, useState } from 'react';
import { Send, Mic, Square } from 'lucide-react';

export default function ChatInputBar({
  onSubmit,
  onVoiceStart,
  onVoiceStop,
  recording = false,
  disabled = false,
}) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);
  const hasText = text.trim().length > 0;

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  }, [text, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit],
  );

  const handleInput = useCallback(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 120) + 'px';
    }
  }, []);

  const handleVoiceToggle = useCallback(() => {
    if (disabled) return;
    if (recording) {
      onVoiceStop?.();
    } else {
      onVoiceStart?.();
    }
  }, [disabled, recording, onVoiceStart, onVoiceStop]);

  return (
    <div className="chat-input-bar">
      <div className={`chat-input-bar__field-wrap${recording ? ' chat-input-bar__field-wrap--recording' : ''}`}>
        <textarea
          ref={textareaRef}
          className="chat-input-bar__textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder={recording ? 'Recording...' : 'Just blurt it out...'}
          rows={1}
          disabled={disabled || recording}
          aria-label="Type a blurt"
        />

        {/* Send button — visible when there's text */}
        <button
          className={`chat-input-bar__send${hasText ? ' chat-input-bar__send--visible' : ''}`}
          onClick={handleSubmit}
          disabled={disabled || !hasText}
          aria-label="Send message"
          type="button"
        >
          <Send size={18} strokeWidth={2.5} />
        </button>
      </div>

      {/* Voice button */}
      <button
        className={`chat-input-bar__voice${recording ? ' chat-input-bar__voice--recording' : ''}`}
        onClick={handleVoiceToggle}
        disabled={disabled}
        aria-label={recording ? 'Stop recording' : 'Start voice input'}
        type="button"
      >
        <span className="chat-input-bar__voice-icon">
          {recording ? <Square size={18} fill="currentColor" /> : <Mic size={20} />}
        </span>
        {recording && <span className="chat-input-bar__voice-pulse" aria-hidden="true" />}
      </button>
    </div>
  );
}
