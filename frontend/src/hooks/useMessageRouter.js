/**
 * useMessageRouter — Discriminates and routes incoming WebSocket messages.
 *
 * Parses each incoming message, classifies it by category
 * (lifecycle / audio_response / push_event / control), and dispatches
 * to the appropriate handler callback. This keeps App.jsx declarative:
 * it registers handlers and the router does the dispatch.
 *
 * Categories:
 *   lifecycle      → session.created, session.updated, session.ended
 *   audio_response → transcript.*, blurt.created, response.text, response.audio, audio.ack
 *   push_event     → task.nudge (and future push types)
 *   control        → error, pong
 */
import { useCallback, useEffect, useRef } from 'react';
import {
  classifyMessage,
  isBinaryAudioFrame,
  MSG_CATEGORY,
  SERVER_MSG,
} from '../utils/messageTypes';

/**
 * @typedef {object} MessageHandlers
 * @property {(payload: object, meta: object) => void} [onTranscriptPartial]
 * @property {(payload: object, meta: object) => void} [onTranscriptFinal]
 * @property {(payload: object, meta: object) => void} [onBlurtCreated]
 * @property {(payload: object, meta: object) => void} [onResponseText]
 * @property {(payload: object, meta: object) => void} [onResponseAudio]
 * @property {(payload: object, meta: object) => void} [onAudioAck]
 * @property {(payload: object, meta: object) => void} [onTaskNudge]
 * @property {(payload: object, meta: object) => void} [onError]
 * @property {(payload: object, meta: object) => void} [onSessionCreated]
 * @property {(payload: object, meta: object) => void} [onSessionEnded]
 * @property {(payload: object, meta: object) => void} [onSessionUpdated]
 * @property {(data: ArrayBuffer|Blob) => void}        [onBinaryAudio]
 */

/**
 * Audio response type → handler key mapping.
 */
const AUDIO_RESPONSE_HANDLERS = Object.freeze({
  [SERVER_MSG.TRANSCRIPT_PARTIAL]: 'onTranscriptPartial',
  [SERVER_MSG.TRANSCRIPT_FINAL]: 'onTranscriptFinal',
  [SERVER_MSG.BLURT_CREATED]: 'onBlurtCreated',
  [SERVER_MSG.RESPONSE_TEXT]: 'onResponseText',
  [SERVER_MSG.RESPONSE_AUDIO]: 'onResponseAudio',
  [SERVER_MSG.AUDIO_ACK]: 'onAudioAck',
});

/**
 * Lifecycle type → handler key mapping.
 */
const LIFECYCLE_HANDLERS = Object.freeze({
  [SERVER_MSG.SESSION_CREATED]: 'onSessionCreated',
  [SERVER_MSG.SESSION_UPDATED]: 'onSessionUpdated',
  [SERVER_MSG.SESSION_ENDED]: 'onSessionEnded',
});

/**
 * Push event type → handler key mapping.
 */
const PUSH_EVENT_HANDLERS = Object.freeze({
  [SERVER_MSG.TASK_NUDGE]: 'onTaskNudge',
});

/**
 * Build a message router callback suitable for useWebSocket(onMessage).
 *
 * @param {MessageHandlers} handlers  Named callbacks for each message type
 * @returns {(rawData: string|ArrayBuffer|Blob) => void}  Callback for useWebSocket
 */
export function useMessageRouter(handlers) {
  // Use a ref so handler identity changes don't re-create the router
  const handlersRef = useRef(handlers);
  useEffect(() => {
    handlersRef.current = handlers;
  }, [handlers]);

  /**
   * Route a parsed JSON message to the correct handler.
   */
  const routeMessage = useCallback((msg) => {
    const h = handlersRef.current;
    if (!h || !msg) return;

    const classified = classifyMessage(msg);
    const { category, type, payload, meta } = classified;

    switch (category) {
      case MSG_CATEGORY.AUDIO_RESPONSE: {
        const handlerKey = AUDIO_RESPONSE_HANDLERS[type];
        if (handlerKey && h[handlerKey]) {
          h[handlerKey](payload, meta);
        }
        break;
      }

      case MSG_CATEGORY.PUSH_EVENT: {
        const handlerKey = PUSH_EVENT_HANDLERS[type];
        if (handlerKey && h[handlerKey]) {
          h[handlerKey](payload, meta);
        }
        break;
      }

      case MSG_CATEGORY.LIFECYCLE: {
        const handlerKey = LIFECYCLE_HANDLERS[type];
        if (handlerKey && h[handlerKey]) {
          h[handlerKey](payload, meta);
        }
        break;
      }

      case MSG_CATEGORY.CONTROL: {
        if (type === SERVER_MSG.ERROR && h.onError) {
          h.onError(payload, meta);
        }
        // pong is handled silently (keepalive)
        break;
      }

      default:
        // Unknown message type — log for debugging in dev
        if (import.meta.env.DEV) {
          console.warn('[MessageRouter] Unknown message type:', type, msg);
        }
        break;
    }
  }, []);

  /**
   * Top-level dispatch that handles both binary and JSON frames.
   * This is the raw onMessage callback wired into useWebSocket.
   */
  const dispatch = useCallback(
    (rawData) => {
      // Binary audio frame (TTS response) — bypass JSON parsing
      if (isBinaryAudioFrame(rawData)) {
        const h = handlersRef.current;
        if (h?.onBinaryAudio) {
          h.onBinaryAudio(rawData);
        }
        return;
      }

      // JSON message — already parsed by useWebSocket before dispatch
      // If it's a string (fallback), parse it; if it's an object, use directly
      if (typeof rawData === 'string') {
        try {
          const msg = JSON.parse(rawData);
          routeMessage(msg);
        } catch {
          if (import.meta.env.DEV) {
            console.warn('[MessageRouter] Unparseable message:', rawData);
          }
        }
      } else if (typeof rawData === 'object' && rawData !== null) {
        // useWebSocket already parses JSON — msg comes as an object
        routeMessage(rawData);
      }
    },
    [routeMessage],
  );

  return dispatch;
}
