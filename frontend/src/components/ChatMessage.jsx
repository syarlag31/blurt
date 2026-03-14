/**
 * Individual message pair in the chat feed.
 *
 * Renders user messages (right-aligned) and server acknowledgment
 * responses (left-aligned) in a chat-style layout.
 *
 * Supports:
 * - Paired rendering: user bubble + ack bubble grouped together
 * - Standalone server messages (nudges, answers)
 * - Emotion-aware ack styling with accent color
 * - Voice badge (SVG mic icon) for voice-captured messages
 * - Intent badge showing classification label on each bubble
 * - Tap-to-expand: collapsed shows text only, expanded reveals
 *   metadata, timestamps, intent badges, and classification details
 * - CSS transition micro-interactions from ui-ux-pro-max design system
 */
import { useState, useCallback, useRef } from 'react';
import { Mic, ChevronDown } from 'lucide-react';
import { MetadataCard } from './MetadataCard';
import { IntentBadge } from './IntentBadge';
import { EventCard } from './EventCard';
import { QuestionAnswerCard } from './QuestionAnswerCard';
import { EMOTION_COLORS } from '../utils/constants';
import { isEventIntent, extractEventData } from '../utils/eventExtractor';
import './ChatMessage.css';

/**
 * Format a timestamp for display.
 * @param {string|number} ts - ISO string or epoch
 * @returns {string} e.g. "2:34 PM"
 */
function formatTime(ts) {
  if (!ts) return '';
  return new Date(ts).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Extract intent string from message or metadata.
 * @param {object} msg - message or ack object
 * @returns {string|null}
 */
function getIntent(msg) {
  if (msg?.metadata?.intent) return msg.metadata.intent;
  if (msg?.intent) return msg.intent;
  return null;
}

/**
 * Single bubble within a ChatMessage.
 * @param {object} props
 * @param {'user'|'server'} props.role
 * @param {string} props.text
 * @param {string} [props.timestamp]
 * @param {boolean} [props.isVoice]
 * @param {string} [props.emotion] - emotion name for accent styling
 * @param {string} [props.tone] - ack tone (warm, gentle, energetic, etc.)
 * @param {string} [props.intent] - intent label for badge
 * @param {boolean} [props.expanded] - whether the parent message is expanded
 * @param {function} [props.onTap] - tap handler for expand/collapse
 */
function Bubble({ role, text, timestamp, isVoice, emotion, tone, intent, expanded, onTap }) {
  const isUser = role === 'user';
  const time = formatTime(timestamp);

  // Emotion-aware accent on server ack bubbles
  const emotionColor = !isUser && emotion ? EMOTION_COLORS[emotion] : null;
  const bubbleStyle = emotionColor
    ? { borderLeftColor: emotionColor, borderLeftWidth: '3px' }
    : undefined;

  // Tone-based modifier class
  const toneClass = !isUser && tone ? `chat-bubble--${tone}` : '';

  return (
    <div
      className={`chat-bubble ${isUser ? 'chat-bubble--user' : 'chat-bubble--server'} ${toneClass} ${expanded ? 'chat-bubble--expanded' : ''}`}
      style={bubbleStyle}
      onClick={onTap}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onTap?.()}
      aria-expanded={expanded}
    >
      {/* Header row — badges, visible when expanded */}
      <div className={`chat-bubble__header ${expanded ? 'chat-bubble__header--visible' : ''}`}>
        {isVoice && (
          <span className="chat-bubble__voice-badge" aria-label="Voice input">
            <Mic size={12} strokeWidth={2.5} />
          </span>
        )}
        {intent && <IntentBadge intent={intent} />}
      </div>

      <p className="chat-bubble__text">{text}</p>

      {/* Expand indicator — subtle chevron */}
      {!expanded && (intent || isVoice || timestamp) && (
        <span className="chat-bubble__expand-hint" aria-hidden="true">
          <ChevronDown size={12} strokeWidth={2} />
        </span>
      )}

      {/* Timestamp — shown on expand */}
      <div className={`chat-bubble__footer ${expanded ? 'chat-bubble__footer--visible' : ''}`}>
        {time && <span className="chat-bubble__time">{time}</span>}
      </div>
    </div>
  );
}

/**
 * ChatMessage renders a single message entry in the feed.
 *
 * A message can be:
 * 1. A user message (right-aligned) with optional paired ack below
 * 2. A standalone server message (left-aligned)
 *
 * Tap-to-expand toggles between:
 * - Collapsed: message text only (clean, minimal)
 * - Expanded: timestamps, intent badges, voice indicator, metadata card
 *
 * @param {object} props
 * @param {object} props.message
 * @param {string} props.message.role - 'user' or 'server'
 * @param {string} props.message.text - display text
 * @param {string} [props.message.timestamp] - ISO timestamp
 * @param {boolean} [props.message.isVoice] - captured via microphone
 * @param {object} [props.message.metadata] - classification result
 * @param {object} [props.message.ack] - paired acknowledgment from server
 * @param {string} [props.message.ack.text] - ack text (8 words max)
 * @param {string} [props.message.ack.timestamp] - when ack arrived
 * @param {string} [props.message.ack.emotion] - detected emotion
 * @param {string} [props.message.ack.tone] - ack tone
 * @param {string} [props.message.emotion] - emotion on server messages
 * @param {string} [props.message.tone] - tone on server messages
 * @param {string} [props.syncState] - Calendar sync state for EVENT messages
 * @param {function} [props.onEventSync] - Callback to sync event to calendar
 * @param {function} [props.onCalendarConnect] - Callback to connect calendar
 * @param {boolean} [props.calendarConnected] - Whether calendar is connected
 */
export function ChatMessage({ message, syncState, onEventSync, onCalendarConnect, calendarConnected }) {
  const [expanded, setExpanded] = useState(false);
  const isUser = message.role === 'user';
  const intent = getIntent(message);
  const msgRef = useRef(null);

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  // Recording indicator — pulsing animated bubble while audio streams
  if (message.isRecording) {
    return (
      <div className="chat-msg chat-msg--user" role="listitem">
        <div className="chat-bubble chat-bubble--user chat-bubble--recording">
          <span className="chat-bubble__voice-badge chat-bubble__voice-badge--recording" aria-label="Recording">
            <Mic size={14} strokeWidth={2.5} />
          </span>
          <p className="chat-bubble__text">
            <span className="recording-dots">Recording</span>
          </p>
        </div>
      </div>
    );
  }

  // Detect EVENT intent and extract event data for EventCard
  const hasEvent = isEventIntent(message);
  const eventData = hasEvent ? extractEventData(message) : null;

  // Detect QUESTION intent — show synthesized answer card
  const isQuestion = intent && intent.toUpperCase() === 'QUESTION';
  // For QUESTION intent: the question text is the user message text,
  // and the WS answer may be in the ack text or a paired server message
  const questionText = isQuestion ? message.text : null;
  const wsAnswerText = isQuestion
    ? (message.ack?.text || (!isUser ? message.text : null))
    : null;

  const hasExpandableContent = !!(
    intent ||
    message.isVoice ||
    message.timestamp ||
    message.metadata ||
    (message.ack && message.ack.metadata) ||
    hasEvent ||
    isQuestion
  );

  return (
    <div
      ref={msgRef}
      className={`chat-msg ${isUser ? 'chat-msg--user' : 'chat-msg--server'} ${expanded ? 'chat-msg--expanded' : ''}`}
      role="listitem"
    >
      {/* Primary bubble */}
      <Bubble
        role={message.role}
        text={message.text}
        timestamp={message.timestamp}
        isVoice={message.isVoice}
        emotion={message.emotion}
        tone={message.tone}
        intent={intent}
        expanded={expanded}
        onTap={hasExpandableContent ? toggleExpanded : undefined}
      />

      {/* Classification metadata card (under user messages) — revealed on expand */}
      <div className={`chat-msg__metadata ${expanded ? 'chat-msg__metadata--visible' : ''}`}>
        {message.metadata && <MetadataCard metadata={message.metadata} />}
      </div>

      {/* Event card — shown for EVENT intent messages (always visible, not gated by expand) */}
      {hasEvent && eventData && (
        <div className="chat-msg__event-card">
          <EventCard
            eventData={eventData}
            syncState={syncState}
            onSync={() => onEventSync?.(message)}
            onConnect={onCalendarConnect}
            calendarConnected={calendarConnected}
          />
        </div>
      )}

      {/* Question answer card — shown for QUESTION intent (always visible, not gated by expand) */}
      {isQuestion && isUser && questionText && (
        <div className="chat-msg__answer-card">
          <QuestionAnswerCard
            questionText={questionText}
            wsAnswerText={wsAnswerText}
            autoFetch={true}
          />
        </div>
      )}

      {/* Paired acknowledgment bubble (under user messages) */}
      {isUser && message.ack && message.ack.text && (
        <div className={`chat-msg__ack ${expanded ? 'chat-msg__ack--expanded' : ''}`}>
          <Bubble
            role="server"
            text={message.ack.text}
            timestamp={message.ack.timestamp}
            emotion={message.ack.emotion}
            tone={message.ack.tone}
            intent={getIntent(message.ack)}
            expanded={expanded}
            onTap={hasExpandableContent ? toggleExpanded : undefined}
          />
          {/* Ack metadata — revealed on expand */}
          <div className={`chat-msg__metadata ${expanded ? 'chat-msg__metadata--visible' : ''}`}>
            {message.ack.metadata && <MetadataCard metadata={message.ack.metadata} />}
          </div>
        </div>
      )}
    </div>
  );
}
