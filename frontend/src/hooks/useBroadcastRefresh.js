/**
 * useBroadcastRefresh — Cross-tab/cross-page refresh trigger using BroadcastChannel.
 *
 * Listens for 'blurt.created' events on a shared channel and increments
 * a counter each time. The counter can be used as a refreshKey to trigger
 * data re-fetches in any consuming component.
 *
 * Also exports a static `notify()` function for the Chat page to call
 * whenever a blurt.created WebSocket message arrives.
 */
import { useState, useEffect } from 'react';

const CHANNEL_NAME = 'blurt-events';

/** Post a blurt.created event to all listeners. */
export function notifyBlurtCreated() {
  try {
    const ch = new BroadcastChannel(CHANNEL_NAME);
    ch.postMessage({ type: 'blurt.created', ts: Date.now() });
    ch.close();
  } catch {
    // BroadcastChannel not supported — silent fallback
  }
}

/**
 * Hook that returns an incrementing counter when blurt.created
 * is received via BroadcastChannel. Use as refreshKey prop.
 */
export function useBroadcastRefresh() {
  const [refreshKey, setRefreshKey] = useState(0);

  useEffect(() => {
    let ch;
    try {
      ch = new BroadcastChannel(CHANNEL_NAME);
      ch.onmessage = (event) => {
        if (event.data?.type === 'blurt.created') {
          setRefreshKey((k) => k + 1);
        }
      };
    } catch {
      // BroadcastChannel not supported — polling-only fallback
    }
    return () => ch?.close();
  }, []);

  return refreshKey;
}
