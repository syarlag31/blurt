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
 * - Voice badge for voice-captured messages
 * - Expandable MetadataCard for classification details
 * - Timestamps on each bubble
 */
import { MetadataCard } from './MetadataCard';
import { EMOTION_COLORS } from '../utils/constants';

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
 * Single bubble within a ChatMessage.
 * @param {object} props
 * @param {'user'|'server'} props.role
 * @param {string} props.text
 * @param {string} [props.timestamp]
 * @param {boolean} [props.isVoice]
 * @param {string} [props.emotion] - emotion name for accent styling
 * @param {string} [props.tone] - ack tone (warm, gentle, energetic, etc.)
 */
function Bubble({ role, text, timestamp, isVoice, emotion, tone }) {
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
      className={`chat-bubble ${isUser ? 'chat-bubble--user' : 'chat-bubble--server'} ${toneClass}`}
      style={bubbleStyle}
    >
      {isVoice && <span className="chat-bubble__voice-badge" aria-label="Voice input">🎤</span>}
      <p className="chat-bubble__text">{text}</p>
      {time && <span className="chat-bubble__time">{time}</span>}
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
 */
export function ChatMessage({ message }) {
  const isUser = message.role === 'user';

  // Recording indicator — pulsing animated bubble while audio streams
  if (message.isRecording) {
    return (
      <div className="chat-msg chat-msg--user" role="listitem">
        <div className="chat-bubble chat-bubble--user chat-bubble--recording">
          <span className="chat-bubble__voice-badge" aria-label="Recording">🎤</span>
          <p className="chat-bubble__text">
            <span className="recording-dots">Recording</span>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`chat-msg ${isUser ? 'chat-msg--user' : 'chat-msg--server'}`}
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
      />

      {/* Classification metadata card (under user messages) */}
      {message.metadata && <MetadataCard metadata={message.metadata} />}

      {/* Paired acknowledgment bubble (under user messages) */}
      {isUser && message.ack && message.ack.text && (
        <div className="chat-msg__ack">
          <Bubble
            role="server"
            text={message.ack.text}
            timestamp={message.ack.timestamp}
            emotion={message.ack.emotion}
            tone={message.ack.tone}
          />
          {message.ack.metadata && <MetadataCard metadata={message.ack.metadata} />}
        </div>
      )}
    </div>
  );
}
