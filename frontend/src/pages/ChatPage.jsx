/**
 * ChatPage — Primary capture + conversation view.
 *
 * Displays the chat feed with message bubbles showing intent badges,
 * text input bar, and voice button. Fetches recent blurts from REST
 * on mount and receives real-time updates via WebSocket.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChatFeed } from '../components/ChatFeed';
import ChatInputBar from '../components/ChatInputBar';
import '../components/ChatInputBar.css';
import { UndoToast } from '../components/UndoToast';
import { useWebSocket } from '../hooks/useWebSocket';
import { useMessageRouter } from '../hooks/useMessageRouter';
import { useVoiceRecorder } from '../hooks/useVoiceRecorder';
import { useAudioPlayback } from '../hooks/useAudioPlayback';
import { notifyBlurtCreated } from '../hooks/useBroadcastRefresh';
import { useEventCalendarSync } from '../hooks/useEventCalendarSync';
import { useCapture } from '../contexts/CaptureContext';
import { API_BASE, USER_ID } from '../utils/constants';

let msgIdCounter = 0;
function nextMsgId() {
  msgIdCounter += 1;
  return `msg-${msgIdCounter}-${Date.now()}`;
}

export default function ChatPage() {
  const [chatMessages, setChatMessages] = useState([]);
  const { enqueueAudio } = useAudioPlayback();
  const { consumeCaptures, subscribe } = useCapture();

  // EVENT intent → Calendar sync with undo window
  const {
    pendingEvent,
    cancelPendingEvent,
    handleBlurtCreated: detectEventIntent,
    UNDO_WINDOW_MS,
  } = useEventCalendarSync();
  const [calendarToastEvent, setCalendarToastEvent] = useState(null);

  // ── Consume FAB captures from other tabs ────────────────────────
  // On mount: pull in any blurts captured while Chat wasn't active
  useEffect(() => {
    const pending = consumeCaptures();
    if (pending.length > 0) {
      setChatMessages((prev) => {
        const newMsgs = [];
        for (const capture of pending) {
          if (capture.forUserMsgId && capture.ackData) {
            // This is an ack for a previous user message — attach it
            const idx = newMsgs.findIndex((m) => m.id === capture.forUserMsgId);
            if (idx >= 0) {
              newMsgs[idx] = { ...newMsgs[idx], ack: capture.ackData, metadata: capture.metadata };
              continue;
            }
            // Also check existing messages
            const existIdx = prev.findIndex((m) => m.id === capture.forUserMsgId);
            if (existIdx >= 0) {
              const updated = [...prev];
              updated[existIdx] = { ...updated[existIdx], ack: capture.ackData, metadata: capture.metadata };
              return [...updated, ...newMsgs];
            }
          }
          newMsgs.push(capture);
        }
        return [...prev, ...newMsgs];
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Subscribe to real-time FAB captures while Chat is active
  useEffect(() => {
    const unsubscribe = subscribe((capture) => {
      setChatMessages((prev) => {
        if (capture.forUserMsgId && capture.ackData) {
          // Ack for a FAB user message — find and attach
          const idx = prev.findIndex((m) => m.id === capture.forUserMsgId);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = { ...updated[idx], ack: capture.ackData, metadata: capture.metadata };
            return updated;
          }
        }
        return [...prev, capture];
      });
    });
    return unsubscribe;
  }, [subscribe]);

  // ── Audio pipeline response handlers ────────────────────────────
  const onResponseText = useCallback((payload) => {
    const serverMsg = {
      id: nextMsgId(),
      role: 'server',
      text: payload.text || 'Got it.',
      timestamp: new Date().toISOString(),
      metadata: payload.metadata || null,
    };

    if (payload.input_text) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: nextMsgId(),
          role: 'user',
          text: payload.input_text,
          timestamp: new Date().toISOString(),
          isVoice: false,
          metadata: payload.classification || null,
        },
        serverMsg,
      ]);
    } else {
      setChatMessages((prev) => [...prev, serverMsg]);
    }
  }, []);

  const onBlurtCreated = useCallback((payload) => {
    // Notify other tabs/pages (Debug, Tasks) to refresh their data
    notifyBlurtCreated();

    // Detect EVENT intent and trigger calendar sync with undo window
    const isEvent = detectEventIntent(payload);
    if (isEvent) {
      // Show the undo toast — message will be set from pendingEvent
      setCalendarToastEvent({
        title: payload.transcript || payload.text || 'New event',
        status: 'pending',
      });
    }

    const ackData = {
      text: payload.acknowledgment || 'Captured.',
      timestamp: new Date().toISOString(),
      emotion: payload.emotion || null,
      tone: payload.tone || null,
      metadata: {
        intent: payload.intent,
        confidence: payload.confidence,
        entities: payload.entities,
        emotion: payload.emotion,
        transcript: payload.transcript,
        all_scores: payload.all_scores,
        latency_ms: payload.latency_ms,
      },
    };

    setChatMessages((prev) => {
      const lastUserIdx = prev.findLastIndex(
        (m) => m.role === 'user' && !m.ack,
      );
      if (lastUserIdx >= 0) {
        const updated = [...prev];
        updated[lastUserIdx] = { ...updated[lastUserIdx], ack: ackData };
        return updated;
      }
      return [
        ...prev,
        {
          id: nextMsgId(),
          role: 'server',
          text: ackData.text,
          timestamp: ackData.timestamp,
          emotion: ackData.emotion,
          tone: ackData.tone,
          metadata: ackData.metadata,
        },
      ];
    });
  }, [detectEventIntent]);

  const onTranscriptFinal = useCallback((payload) => {
    if (payload.transcript) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: nextMsgId(),
          role: 'user',
          text: payload.transcript,
          timestamp: new Date().toISOString(),
          isVoice: true,
          metadata: payload.classification || null,
        },
      ]);
    }
  }, []);

  const onResponseAudio = useCallback(
    (payload) => {
      if (payload.audio_data) enqueueAudio(payload.audio_data);
    },
    [enqueueAudio],
  );

  const onTaskNudge = useCallback(() => {}, []);
  const onError = useCallback((payload) => {
    if (import.meta.env.DEV) {
      console.warn('[Blurt] Server error:', payload.message || payload.code);
    }
  }, []);
  const onBinaryAudio = useCallback(
    (data) => enqueueAudio(data),
    [enqueueAudio],
  );

  const messageHandlers = useMemo(
    () => ({
      onResponseText,
      onBlurtCreated,
      onTranscriptFinal,
      onResponseAudio,
      onTaskNudge,
      onError,
      onBinaryAudio,
    }),
    [onResponseText, onBlurtCreated, onTranscriptFinal, onResponseAudio, onTaskNudge, onError, onBinaryAudio],
  );

  const routeMessage = useMessageRouter(messageHandlers);
  const { connected, sendBinary, sendTextInput, sendAudioCommit } =
    useWebSocket(routeMessage, onBinaryAudio);
  const { recording, startRecording, stopRecording } =
    useVoiceRecorder({ sendBinary, sendAudioCommit });

  // Fetch recent episodes on mount to populate chat history.
  // Tries /episodes/user/{id} (canonical) with /blurts fallback.
  useEffect(() => {
    const fetchRecent = async () => {
      try {
        // Primary: episodes endpoint returns EpisodeResponse objects
        let res = await fetch(
          `${API_BASE}/episodes/user/${encodeURIComponent(USER_ID)}?limit=50`,
        );
        let episodes = [];
        if (res.ok) {
          const data = await res.json();
          episodes = data.episodes || [];
        } else {
          // Fallback: /blurts endpoint (legacy)
          res = await fetch(
            `${API_BASE}/blurts?user_id=${encodeURIComponent(USER_ID)}&limit=50`,
          );
          if (!res.ok) return;
          const data = await res.json();
          episodes = data.blurts || data.items || (Array.isArray(data) ? data : []);
        }

        if (episodes.length > 0) {
          const msgs = episodes.map((ep) => {
            // Normalize field names across endpoint variants:
            // Episodes API: raw_text, modality, intent_confidence, emotion (object)
            // Legacy /blurts: content/transcript/text, source, confidence
            const text = ep.raw_text || ep.content || ep.transcript || ep.text || '';
            const intent = ep.intent || null;
            const confidence = ep.intent_confidence ?? ep.confidence ?? null;
            const entities = ep.entities || [];
            const emotion = ep.emotion || null;
            const emotionLabel = typeof emotion === 'string' ? emotion : emotion?.primary || null;
            const isVoice = ep.modality === 'voice' || ep.source === 'voice' || ep.source === 'audio';
            const timestamp = ep.timestamp || ep.created_at || new Date().toISOString();

            // Acknowledgment — may be string or AcknowledgmentResponse object
            const ackText = typeof ep.acknowledgment === 'string'
              ? ep.acknowledgment
              : ep.acknowledgment?.text || null;
            const ackTone = typeof ep.acknowledgment === 'object'
              ? ep.acknowledgment?.tone
              : ep.tone || null;

            return {
              id: ep.id || ep.blurt_id || nextMsgId(),
              role: 'user',
              text,
              timestamp,
              isVoice,
              emotion: emotionLabel,
              metadata: intent ? {
                intent,
                confidence,
                entities,
                emotion,
                all_scores: ep.all_scores || null,
              } : null,
              ack: ackText ? {
                text: ackText,
                timestamp,
                emotion: emotionLabel,
                tone: ackTone,
                metadata: {
                  intent,
                  confidence,
                  entities,
                  emotion,
                },
              } : null,
            };
          });
          setChatMessages(msgs);
        }
      } catch {
        // API not available — fine for dogfooding
      }
    };
    fetchRecent();
  }, []);

  // Sync calendar toast state with pending event from hook
  useEffect(() => {
    if (pendingEvent) {
      setCalendarToastEvent({
        title: pendingEvent.title,
        status: pendingEvent.status,
      });
    } else if (calendarToastEvent?.status === 'pending') {
      // pendingEvent cleared = either synced or cancelled
      // The hook handles history; we just need to dismiss the toast after a beat
    }
  }, [pendingEvent]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleUndoCalendarEvent = useCallback(() => {
    cancelPendingEvent();
    setCalendarToastEvent((prev) =>
      prev ? { ...prev, status: 'cancelled' } : null,
    );
  }, [cancelPendingEvent]);

  const handleDismissCalendarToast = useCallback(() => {
    setCalendarToastEvent(null);
  }, []);

  // Append recording indicator when active
  const displayMessages = useMemo(() => {
    if (!recording) return chatMessages;
    return [
      ...chatMessages,
      {
        id: 'recording-indicator',
        role: 'user',
        text: 'Recording...',
        timestamp: new Date().toISOString(),
        isVoice: true,
        isRecording: true,
      },
    ];
  }, [chatMessages, recording]);

  // Reconnection notice
  const prevConnected = useRef(connected);
  useEffect(() => {
    if (prevConnected.current && !connected && chatMessages.length > 0) {
      setChatMessages((prev) => [
        ...prev,
        {
          id: nextMsgId(),
          role: 'server',
          text: 'Reconnecting...',
          timestamp: new Date().toISOString(),
        },
      ]);
    }
    prevConnected.current = connected;
  }, [connected, chatMessages.length]);

  // Text submit handler — WS first, REST fallback
  const handleTextSubmit = useCallback(
    async (text) => {
      setChatMessages((prev) => [
        ...prev,
        {
          id: nextMsgId(),
          role: 'user',
          text,
          timestamp: new Date().toISOString(),
          isVoice: false,
        },
      ]);

      if (connected) {
        sendTextInput(text);
      } else {
        try {
          const res = await fetch(`${API_BASE}/capture/text`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: USER_ID, text }),
          });
          if (res.ok) {
            const data = await res.json();
            // BlurtResponse: { acknowledgment: {text, tone}, intent, intent_confidence,
            //   episode: {entities, emotion, ...}, entities_extracted, ... }
            const ackText = typeof data.acknowledgment === 'string'
              ? data.acknowledgment
              : data.acknowledgment?.text || data.ack || 'Captured.';
            const ackTone = typeof data.acknowledgment === 'object'
              ? data.acknowledgment?.tone : null;
            const ep = data.episode || {};
            setChatMessages((prev) => {
              const updated = [...prev];
              // Attach metadata to the last user message
              const lastIdx = updated.findLastIndex((m) => m.role === 'user' && !m.ack);
              if (lastIdx >= 0) {
                updated[lastIdx] = {
                  ...updated[lastIdx],
                  metadata: {
                    intent: data.intent || ep.intent,
                    confidence: data.intent_confidence ?? ep.intent_confidence ?? null,
                    entities: ep.entities || [],
                    emotion: ep.emotion || null,
                  },
                  ack: {
                    text: ackText,
                    timestamp: new Date().toISOString(),
                    emotion: typeof ep.emotion === 'string' ? ep.emotion : ep.emotion?.primary || null,
                    tone: ackTone,
                    metadata: {
                      intent: data.intent || ep.intent,
                      confidence: data.intent_confidence ?? ep.intent_confidence ?? null,
                      entities: ep.entities || [],
                      emotion: ep.emotion || null,
                    },
                  },
                };
                return updated;
              }
              // Fallback: standalone server message
              return [
                ...prev,
                {
                  id: nextMsgId(),
                  role: 'server',
                  text: ackText,
                  timestamp: new Date().toISOString(),
                  metadata: {
                    intent: data.intent || ep.intent,
                    confidence: data.intent_confidence ?? ep.intent_confidence ?? null,
                    entities: ep.entities || [],
                    emotion: ep.emotion || null,
                  },
                },
              ];
            });
          }
        } catch {
          setChatMessages((prev) => [
            ...prev,
            {
              id: nextMsgId(),
              role: 'server',
              text: 'Connection lost. Try again.',
              timestamp: new Date().toISOString(),
            },
          ]);
        }
      }
    },
    [connected, sendTextInput],
  );

  const handleVoiceStart = useCallback(() => {
    startRecording();
  }, [startRecording]);

  const handleVoiceStop = useCallback(() => {
    stopRecording();
  }, [stopRecording]);

  // Build undo toast message for calendar events
  const calendarUndoMessage = calendarToastEvent
    ? calendarToastEvent.status === 'cancelled'
      ? 'Event cancelled'
      : `Adding to calendar: ${calendarToastEvent.title?.slice(0, 40) || 'Event'}${(calendarToastEvent.title?.length || 0) > 40 ? '...' : ''}`
    : '';

  return (
    <div className="page page--chat page--chat-layout">
      <div className="page--chat__feed">
        <ChatFeed chatMessages={displayMessages} />
      </div>

      <ChatInputBar
        onSubmit={handleTextSubmit}
        onVoiceStart={handleVoiceStart}
        onVoiceStop={handleVoiceStop}
        recording={recording}
        disabled={!connected}
      />

      {/* Calendar sync undo toast — shown when EVENT intent detected */}
      {calendarToastEvent && (
        <UndoToast
          key={calendarToastEvent.title}
          message={calendarUndoMessage}
          onUndo={handleUndoCalendarEvent}
          onDismiss={handleDismissCalendarToast}
          duration={UNDO_WINDOW_MS}
        />
      )}
    </div>
  );
}
