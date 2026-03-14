/**
 * CaptureContext — Shared capture state bridging FAB blurts to Chat feed.
 *
 * When a blurt is captured via the FAB on any non-Chat tab, it:
 * 1. Sends the text to the backend via REST POST /capture/text
 * 2. Stores the user message + server response in a shared queue
 * 3. ChatPage subscribes and incorporates pending captures into its feed
 *
 * This ensures blurts captured from Memory, Tasks, or Debug tabs
 * appear in the Chat feed seamlessly.
 */
import { createContext, useCallback, useContext, useRef, useState } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';
import { notifyBlurtCreated } from '../hooks/useBroadcastRefresh';

const CaptureContext = createContext(null);

let captureIdCounter = 0;
function nextCaptureId() {
  captureIdCounter += 1;
  return `fab-capture-${captureIdCounter}-${Date.now()}`;
}

/**
 * Provider component — wrap around AppShell so both FAB and ChatPage
 * share the same capture queue.
 */
export function CaptureProvider({ children }) {
  // Queue of captures from FAB that ChatPage hasn't consumed yet
  const [pendingCaptures, setPendingCaptures] = useState([]);
  // Subscribers that want to be notified when new captures arrive
  const subscribersRef = useRef([]);

  /**
   * Subscribe to new FAB captures. Returns unsubscribe function.
   * Called by ChatPage to get real-time updates.
   */
  const subscribe = useCallback((callback) => {
    subscribersRef.current.push(callback);
    return () => {
      subscribersRef.current = subscribersRef.current.filter(
        (cb) => cb !== callback,
      );
    };
  }, []);

  /**
   * Capture a text blurt via REST API. Stores the user message
   * and server response for ChatPage to pick up.
   */
  const captureBlurt = useCallback(async (text) => {
    const userMsg = {
      id: nextCaptureId(),
      role: 'user',
      text,
      timestamp: new Date().toISOString(),
      isVoice: false,
      fromFAB: true,
    };

    // Immediately push user message to pending queue
    setPendingCaptures((prev) => [...prev, userMsg]);
    // Notify subscribers (ChatPage) immediately
    subscribersRef.current.forEach((cb) => cb(userMsg));

    try {
      const res = await fetch(`${API_BASE}/capture/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: USER_ID, text }),
      });

      if (res.ok) {
        const data = await res.json();
        const ackText =
          typeof data.acknowledgment === 'string'
            ? data.acknowledgment
            : data.acknowledgment?.text || data.ack || 'Captured.';
        const ackTone =
          typeof data.acknowledgment === 'object'
            ? data.acknowledgment?.tone
            : null;
        const ep = data.episode || {};

        const ackMsg = {
          id: nextCaptureId(),
          role: 'server',
          text: ackText,
          timestamp: new Date().toISOString(),
          metadata: {
            intent: data.intent || ep.intent,
            confidence:
              data.intent_confidence ?? ep.intent_confidence ?? null,
            entities: ep.entities || [],
            emotion: ep.emotion || null,
          },
          // Attach ack data to user message lookup
          forUserMsgId: userMsg.id,
          ackData: {
            text: ackText,
            timestamp: new Date().toISOString(),
            emotion:
              typeof ep.emotion === 'string'
                ? ep.emotion
                : ep.emotion?.primary || null,
            tone: ackTone,
            metadata: {
              intent: data.intent || ep.intent,
              confidence:
                data.intent_confidence ?? ep.intent_confidence ?? null,
              entities: ep.entities || [],
              emotion: ep.emotion || null,
            },
          },
        };

        setPendingCaptures((prev) => [...prev, ackMsg]);
        subscribersRef.current.forEach((cb) => cb(ackMsg));

        // Notify other tabs (Debug, Tasks, Memory) to refresh
        notifyBlurtCreated();
      }
    } catch {
      // API not available — the user message still appears in chat
      const errorMsg = {
        id: nextCaptureId(),
        role: 'server',
        text: 'Sent (offline).',
        timestamp: new Date().toISOString(),
        fromFAB: true,
      };
      setPendingCaptures((prev) => [...prev, errorMsg]);
      subscribersRef.current.forEach((cb) => cb(errorMsg));
    }
  }, []);

  /**
   * Consume and clear all pending captures. Called by ChatPage
   * after incorporating them into its chat messages state.
   */
  const consumeCaptures = useCallback(() => {
    let captured;
    setPendingCaptures((prev) => {
      captured = prev;
      return [];
    });
    return captured || [];
  }, []);

  const value = {
    pendingCaptures,
    captureBlurt,
    consumeCaptures,
    subscribe,
  };

  return (
    <CaptureContext.Provider value={value}>{children}</CaptureContext.Provider>
  );
}

/**
 * Hook to access capture context.
 */
export function useCapture() {
  const ctx = useContext(CaptureContext);
  if (!ctx) {
    throw new Error('useCapture must be used within a CaptureProvider');
  }
  return ctx;
}
