/**
 * Text input bar for typed blurts.
 *
 * Mobile-friendly with large touch target, auto-expanding textarea,
 * and submit on Enter (shift+Enter for newline).
 */
import { useCallback, useRef, useState } from 'react';

export function TextInput({ onSubmit, disabled }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSubmit(trimmed);
    setText('');
    // Reset textarea height
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

  return (
    <div className="text-input">
      <textarea
        ref={textareaRef}
        className="text-input__field"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        placeholder="Just blurt it out..."
        rows={1}
        disabled={disabled}
        aria-label="Type a blurt"
      />
      <button
        className="text-input__send"
        onClick={handleSubmit}
        disabled={disabled || !text.trim()}
        aria-label="Send"
      >
        ↑
      </button>
    </div>
  );
}
