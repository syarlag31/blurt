/**
 * Server WebSocket message types and category classification.
 *
 * Maps every ServerMessageType (from blurt/models/audio.py) to a
 * routing category so the frontend can dispatch incoming messages
 * to the correct handler without a sprawling switch statement.
 */

// ── Server → Client message type constants ──────────────────────────
export const SERVER_MSG = Object.freeze({
  // Lifecycle
  SESSION_CREATED: 'session.created',
  SESSION_UPDATED: 'session.updated',
  SESSION_ENDED: 'session.ended',

  // Audio pipeline responses
  AUDIO_ACK: 'audio.ack',
  TRANSCRIPT_PARTIAL: 'transcript.partial',
  TRANSCRIPT_FINAL: 'transcript.final',
  BLURT_CREATED: 'blurt.created',
  RESPONSE_TEXT: 'response.text',
  RESPONSE_AUDIO: 'response.audio',

  // Server-push events
  TASK_NUDGE: 'task.nudge',

  // Control
  ERROR: 'error',
  PONG: 'pong',
});

// ── Message categories ──────────────────────────────────────────────
export const MSG_CATEGORY = Object.freeze({
  /** Session lifecycle (created, updated, ended) */
  LIFECYCLE: 'lifecycle',
  /** Audio pipeline results (transcripts, acks, blurt confirmations) */
  AUDIO_RESPONSE: 'audio_response',
  /** Server-initiated push events (task nudges, notifications) */
  PUSH_EVENT: 'push_event',
  /** Control messages (ping/pong, errors) */
  CONTROL: 'control',
  /** Unknown / unrecognized message type */
  UNKNOWN: 'unknown',
});

/**
 * Mapping from message type → category.
 * Used by classifyMessage() for O(1) routing.
 */
const TYPE_TO_CATEGORY = Object.freeze({
  // Lifecycle
  [SERVER_MSG.SESSION_CREATED]: MSG_CATEGORY.LIFECYCLE,
  [SERVER_MSG.SESSION_UPDATED]: MSG_CATEGORY.LIFECYCLE,
  [SERVER_MSG.SESSION_ENDED]: MSG_CATEGORY.LIFECYCLE,

  // Audio pipeline responses
  [SERVER_MSG.AUDIO_ACK]: MSG_CATEGORY.AUDIO_RESPONSE,
  [SERVER_MSG.TRANSCRIPT_PARTIAL]: MSG_CATEGORY.AUDIO_RESPONSE,
  [SERVER_MSG.TRANSCRIPT_FINAL]: MSG_CATEGORY.AUDIO_RESPONSE,
  [SERVER_MSG.BLURT_CREATED]: MSG_CATEGORY.AUDIO_RESPONSE,
  [SERVER_MSG.RESPONSE_TEXT]: MSG_CATEGORY.AUDIO_RESPONSE,
  [SERVER_MSG.RESPONSE_AUDIO]: MSG_CATEGORY.AUDIO_RESPONSE,

  // Server-push events
  [SERVER_MSG.TASK_NUDGE]: MSG_CATEGORY.PUSH_EVENT,

  // Control
  [SERVER_MSG.ERROR]: MSG_CATEGORY.CONTROL,
  [SERVER_MSG.PONG]: MSG_CATEGORY.CONTROL,
});

/**
 * Classify a parsed WebSocket message into a routing category.
 *
 * @param {object} msg  Parsed JSON message with a `type` field
 * @returns {{ category: string, type: string, payload: object, meta: object }}
 */
export function classifyMessage(msg) {
  const type = msg?.type ?? '';
  const category = TYPE_TO_CATEGORY[type] ?? MSG_CATEGORY.UNKNOWN;

  return {
    category,
    type,
    payload: msg?.payload ?? {},
    meta: {
      sessionId: msg?.session_id ?? null,
      sequence: msg?.sequence ?? 0,
      timestamp: msg?.timestamp ?? null,
    },
  };
}

/**
 * Check if a message is a binary audio response frame.
 * Binary frames arrive as ArrayBuffer/Blob on the WebSocket —
 * they skip JSON parsing entirely.
 *
 * @param {*} data  Raw WebSocket event.data
 * @returns {boolean}
 */
export function isBinaryAudioFrame(data) {
  return data instanceof ArrayBuffer || data instanceof Blob;
}
