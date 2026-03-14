/**
 * ChatFeed — scrollable container rendering ChatMessage + MetadataCard pairs.
 *
 * Features:
 * - Auto-scrolls to bottom when new messages arrive (if user is near bottom)
 * - Shows "New messages ↓" pill when user has scrolled up and new messages arrive
 * - Smooth scroll-to-bottom on pill tap
 * - Empty state placeholder when no messages exist
 * - Connects to parent state (driven by WebSocket via App.jsx)
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { ChatMessage } from './ChatMessage';

/** How close to the bottom (in px) the user must be to trigger auto-scroll */
const SCROLL_THRESHOLD = 120;

export function ChatFeed({ chatMessages }) {
  const feedRef = useRef(null);
  const bottomRef = useRef(null);
  const [hasNewMessages, setHasNewMessages] = useState(false);
  const prevLengthRef = useRef(chatMessages.length);
  const nearBottomRef = useRef(true);

  // Track whether user is near the bottom of the scroll container
  const handleScroll = useCallback(() => {
    const el = feedRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    nearBottomRef.current = distanceFromBottom <= SCROLL_THRESHOLD;
    if (nearBottomRef.current) {
      setHasNewMessages(false);
    }
  }, []);

  // When new messages arrive, auto-scroll if near bottom; otherwise show indicator
  // Uses a ref callback pattern to avoid setState in effect body
  useEffect(() => {
    const newCount = chatMessages.length;
    const prevCount = prevLengthRef.current;
    prevLengthRef.current = newCount;

    if (newCount <= prevCount) return; // no new messages

    if (nearBottomRef.current) {
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      });
    } else {
      // Schedule the state update outside the synchronous effect body
      queueMicrotask(() => setHasNewMessages(true));
    }
  }, [chatMessages.length]);

  // Scroll to bottom on initial mount
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'instant' });
  }, []);

  // Scroll to bottom when user taps the "new messages" pill
  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    setHasNewMessages(false);
  }, []);

  // Empty state
  if (chatMessages.length === 0) {
    return (
      <div className="chat-feed chat-feed--empty" ref={feedRef}>
        <div className="chat-feed__placeholder">
          <p className="chat-feed__placeholder-emoji">💭</p>
          <p className="chat-feed__placeholder-text">
            Just blurt it out.
          </p>
          <p className="chat-feed__placeholder-hint">
            Tap the mic or type below
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-feed" ref={feedRef} onScroll={handleScroll} role="list" aria-label="Chat messages">
      {chatMessages.map((msg, i) => (
        <ChatMessage key={msg.id || i} message={msg} />
      ))}
      <div ref={bottomRef} aria-hidden="true" />

      {hasNewMessages && (
        <button
          className="chat-feed__new-msg-pill"
          onClick={scrollToBottom}
          aria-label="Scroll to new messages"
        >
          New messages ↓
        </button>
      )}
    </div>
  );
}
