/**
 * Persistent WebSocket hook for Blurt.
 *
 * Maintains a single WebSocket connection that stays alive across
 * multiple audio capture sessions. Supports:
 *   - Multiple session.init → session.end cycles
 *   - Server-push messages (task nudges) at any time
 *   - Keepalive pings between capture sessions
 *   - Automatic reconnection with exponential backoff
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { USER_ID, WS_URL } from '../utils/constants';

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 30000;

/**
 * @param {(msg: object) => void} onMessage — called for each parsed WS message
 * @param {(data: ArrayBuffer) => void} [onBinary] — called for each binary WS frame (audio response)
 */
export function useWebSocket(onMessage, onBinary) {
  const wsRef = useRef(null);
  const sequenceRef = useRef(0);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef(null);
  const intentionalClose = useRef(false);
  const connectRef = useRef(null);
  const onMessageRef = useRef(onMessage);

  const [connected, setConnected] = useState(false);
  const [sessionReady, setSessionReady] = useState(false);

  const onBinaryRef = useRef(onBinary);

  // Keep the callback refs fresh without re-creating the WS
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    onBinaryRef.current = onBinary;
  }, [onBinary]);

  const send = useCallback((msg) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  }, []);

  const sendBinary = useCallback((data) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(data);
    }
  }, []);

  const nextSequence = useCallback(() => {
    sequenceRef.current += 1;
    return sequenceRef.current;
  }, []);

  // Connection setup — single persistent lifecycle
  useEffect(() => {
    function doConnect() {
      if (wsRef.current?.readyState === WebSocket.OPEN) return;

      intentionalClose.current = false;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectAttempt.current = 0;
        sequenceRef.current = 0;

        // Send session.init immediately on connect
        const seq = sequenceRef.current + 1;
        sequenceRef.current = seq;
        ws.send(
          JSON.stringify({
            type: 'session.init',
            sequence: seq,
            payload: {
              user_id: USER_ID,
              audio_config: {
                encoding: 'opus',
                sample_rate: 48000,
                channels: 1,
                language: 'en-US',
              },
              timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
            },
          }),
        );
      };

      ws.binaryType = 'arraybuffer';

      ws.onmessage = (event) => {
        // Handle binary frames (server-sent audio)
        if (event.data instanceof ArrayBuffer) {
          onBinaryRef.current?.(event.data);
          return;
        }

        try {
          const msg = JSON.parse(event.data);

          // Track session lifecycle on the persistent connection
          if (msg.type === 'session.created') {
            setSessionReady(true);
          } else if (msg.type === 'session.ended') {
            // Session ended but connection stays alive
            setSessionReady(false);
          }

          // Dispatch to consumer callback (handles task.nudge, etc.)
          onMessageRef.current?.(msg);
        } catch {
          // Unparseable text — ignore
        }
      };

      ws.onclose = () => {
        setConnected(false);
        setSessionReady(false);
        wsRef.current = null;

        if (!intentionalClose.current) {
          const delay = Math.min(
            RECONNECT_BASE_MS * Math.pow(2, reconnectAttempt.current),
            RECONNECT_MAX_MS,
          );
          reconnectAttempt.current += 1;
          reconnectTimer.current = setTimeout(() => connectRef.current?.(), delay);
        }
      };

      ws.onerror = () => {
        // onclose fires after onerror — reconnect handled there
      };
    }

    connectRef.current = doConnect;
    doConnect();

    return () => {
      intentionalClose.current = true;
      clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []); // Single connection lifecycle — never re-creates

  // Re-initialize a session on the existing persistent connection
  const reinitSession = useCallback(() => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    const seq = nextSequence();
    ws.send(
      JSON.stringify({
        type: 'session.init',
        sequence: seq,
        payload: {
          user_id: USER_ID,
          audio_config: {
            encoding: 'opus',
            sample_rate: 48000,
            channels: 1,
            language: 'en-US',
          },
          timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        },
      }),
    );
  }, [nextSequence]);

  // End the current capture session (connection stays alive)
  const endSession = useCallback(() => {
    send({
      type: 'session.end',
      sequence: nextSequence(),
      payload: {},
    });
  }, [send, nextSequence]);

  // Send text input as a blurt (works with or without active session)
  const sendTextInput = useCallback(
    (text) => {
      send({
        type: 'text.input',
        sequence: nextSequence(),
        payload: { text },
      });
    },
    [send, nextSequence],
  );

  // Send audio commit (end of utterance)
  const sendAudioCommit = useCallback(() => {
    send({
      type: 'audio.commit',
      sequence: nextSequence(),
      payload: {},
    });
  }, [send, nextSequence]);

  // Keepalive ping — runs regardless of session state
  useEffect(() => {
    if (!connected) return;
    const interval = setInterval(() => {
      send({ type: 'ping', sequence: 0 });
    }, 25000);
    return () => clearInterval(interval);
  }, [connected, send]);

  return {
    connected,
    sessionReady,
    send,
    sendBinary,
    sendTextInput,
    sendAudioCommit,
    reinitSession,
    endSession,
    nextSequence,
  };
}
