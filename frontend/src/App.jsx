/**
 * Blurt — Main application component.
 *
 * Mobile-first layout:
 * ┌──────────────────────┐
 * │  Header + Status     │
 * │  TaskCard (sticky)   │
 * │  ┌────────────────┐  │
 * │  │ Chat Feed      │  │
 * │  │ (scrollable)   │  │
 * │  └────────────────┘  │
 * │  Input Bar + Voice   │
 * └──────────────────────┘
 *
 * Message routing: useMessageRouter classifies every incoming WebSocket
 * message by category (audio_response | push_event | lifecycle | control)
 * and dispatches to the matching handler registered below. This replaces
 * the previous inline switch with structured type discrimination.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ChatFeed } from './components/ChatFeed';
import { ConnectionStatus } from './components/ConnectionStatus';
import { NudgeToast } from './components/NudgeToast';
import { TaskCard } from './components/TaskCard';
import { TextInput } from './components/TextInput';
import { VoiceButton } from './components/VoiceButton';
import { useAudioPlayback } from './hooks/useAudioPlayback';
import { useMessageRouter } from './hooks/useMessageRouter';
import { useVoiceRecorder } from './hooks/useVoiceRecorder';
import { useWebSocket } from './hooks/useWebSocket';
import { API_BASE, USER_ID } from './utils/constants';
import './App.css';

let msgIdCounter = 0;
function nextMsgId() {
  msgIdCounter += 1;
  return `msg-${msgIdCounter}-${Date.now()}`;
}

export default function App() {
  // Chat messages displayed in the feed
  const [chatMessages, setChatMessages] = useState([]);
  // Current top task from surfacing
  const [topTask, setTopTask] = useState(null);
  // Whether the current task arrived via a server-push nudge
  const [isNudge, setIsNudge] = useState(false);
  // Active nudge for toast notification (cleared after auto-dismiss)
  const [activeNudge, setActiveNudge] = useState(null);
  // Nudge counter to uniquely key toasts even for same-content tasks
  const nudgeCounter = useRef(0);
  // Audio playback for server-sent audio responses
  const { playing: audioPlaying, enqueueAudio, stopPlayback } = useAudioPlayback();

  // ── Audio pipeline response handlers ────────────────────────────
  // Dispatched by useMessageRouter for MSG_CATEGORY.AUDIO_RESPONSE

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
      // Find the last user message without an ack to pair with
      const lastUserIdx = prev.findLastIndex(
        (m) => m.role === 'user' && !m.ack,
      );
      if (lastUserIdx >= 0) {
        // Pair the ack with the user message
        const updated = [...prev];
        updated[lastUserIdx] = { ...updated[lastUserIdx], ack: ackData };
        return updated;
      }
      // No user message to pair — show as standalone server message
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
  }, []);

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
      // Audio response from server (base64-encoded audio data in payload)
      if (payload.audio_data) {
        enqueueAudio(payload.audio_data);
      }
    },
    [enqueueAudio],
  );

  // ── Push event handlers ─────────────────────────────────────────
  // Dispatched by useMessageRouter for MSG_CATEGORY.PUSH_EVENT

  const onTaskNudge = useCallback((payload) => {
    if (payload.task) {
      nudgeCounter.current += 1;
      const key = nudgeCounter.current;
      setTopTask({ ...payload.task, _nudgeKey: key });
      setIsNudge(true);
      setActiveNudge({ ...payload.task, _nudgeKey: key });
    }
  }, []);

  // ── Control handlers ────────────────────────────────────────────
  // Dispatched by useMessageRouter for MSG_CATEGORY.CONTROL

  const onError = useCallback((payload) => {
    const errorMsg = payload.message || payload.code || 'Unknown error';
    if (import.meta.env.DEV) {
      console.warn('[Blurt] Server error:', errorMsg);
    }
  }, []);

  // ── Binary audio handler (raw ArrayBuffer from server TTS) ──────
  const onBinaryAudio = useCallback(
    (data) => {
      enqueueAudio(data);
    },
    [enqueueAudio],
  );

  // ── Wire all handlers into the message router ───────────────────
  // useMessageRouter classifies each message and dispatches to the
  // matching handler by category + type. See utils/messageTypes.js
  // for the full type-to-category mapping.
  const messageHandlers = useMemo(
    () => ({
      // Audio pipeline responses (MSG_CATEGORY.AUDIO_RESPONSE)
      onResponseText,
      onBlurtCreated,
      onTranscriptFinal,
      onResponseAudio,
      // Server-push events (MSG_CATEGORY.PUSH_EVENT)
      onTaskNudge,
      // Control messages (MSG_CATEGORY.CONTROL)
      onError,
      // Binary audio frames (ArrayBuffer bypass)
      onBinaryAudio,
    }),
    [onResponseText, onBlurtCreated, onTranscriptFinal, onResponseAudio, onTaskNudge, onError, onBinaryAudio],
  );

  const routeMessage = useMessageRouter(messageHandlers);

  // useWebSocket passes parsed JSON to routeMessage, binary to onBinaryAudio
  const {
    connected,
    sessionReady,
    sendBinary,
    sendTextInput,
    sendAudioCommit,
  } = useWebSocket(routeMessage, onBinaryAudio);

  const { recording, startRecording, stopRecording, toggleRecording, error: voiceError } =
    useVoiceRecorder({ sendBinary, sendAudioCommit });

  // Derive the display messages by appending a recording indicator when active.
  // This avoids calling setState inside an effect (which causes cascading renders).
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

  // Fetch initial top task on mount via Thompson Sampling surfacing API
  useEffect(() => {
    const fetchTopTask = async () => {
      try {
        const res = await fetch(
          `${API_BASE}/tasks/surface?user_id=${encodeURIComponent(USER_ID)}&max_results=1`,
        );
        if (res.ok) {
          const data = await res.json();
          if (data.tasks && data.tasks.length > 0) {
            const t = data.tasks[0];
            setTopTask({
              id: t.task_id,
              content: t.content,
              intent: t.intent,
              score: t.composite_score,
              reason: t.surfacing_reason,
              status: t.status,
              entity_names: t.entity_names,
              project: t.project,
            });
          }
        }
      } catch {
        // API not available yet — fine for dogfooding
      }
    };
    fetchTopTask();
  }, []);

  // Handle text submission
  const handleTextSubmit = useCallback(
    (text) => {
      // Optimistically add user message
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
      sendTextInput(text);
    },
    [sendTextInput],
  );

  // Handle task actions (complete/defer/drop) via feedback API
  // Maps UI actions to the FeedbackAction enum on the backend
  const handleTaskAction = useCallback(
    async (action, taskId) => {
      try {
        await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: USER_ID,
            action: action,
          }),
        });
      } catch {
        // Best effort — still clear card for responsive feel
      }
      // Clear the task card and fetch the next best task
      setTopTask(null);
      setIsNudge(false);
      // Fetch next top task after a brief delay for scoring to settle
      setTimeout(async () => {
        try {
          const res = await fetch(
            `${API_BASE}/tasks/surface?user_id=${encodeURIComponent(USER_ID)}&max_results=1`,
          );
          if (res.ok) {
            const data = await res.json();
            if (data.tasks && data.tasks.length > 0) {
              const t = data.tasks[0];
              setTopTask({
                id: t.task_id,
                content: t.content,
                intent: t.intent,
                score: t.composite_score,
                reason: t.surfacing_reason,
                status: t.status,
                entity_names: t.entity_names,
                project: t.project,
              });
            }
          }
        } catch {
          // No next task — that's fine
        }
      }, 300);
    },
    [],
  );

  const inputDisabled = !connected || !sessionReady;

  return (
    <div className="app">
      <header className="app__header">
        <h1 className="app__title">Blurt</h1>
        <ConnectionStatus connected={connected} sessionReady={sessionReady} />
      </header>

      <div className="app__task-area">
        <TaskCard
          task={topTask}
          onAction={handleTaskAction}
          isNudge={isNudge}
          key={topTask?._nudgeKey || 'static'}
        />
        <NudgeToast
          nudge={activeNudge}
          key={activeNudge?._nudgeKey}
          onDismiss={() => {
            setActiveNudge(null);
            setIsNudge(false);
          }}
        />
      </div>

      <main className="app__chat">
        <ChatFeed chatMessages={displayMessages} />
      </main>

      <footer className="app__input-bar">
        {voiceError && <p className="app__voice-error">{voiceError}</p>}
        {audioPlaying && (
          <div className="audio-playback-bar">
            <span className="audio-playback-bar__visualizer">
              <span className="audio-playback-bar__bar" />
              <span className="audio-playback-bar__bar" />
              <span className="audio-playback-bar__bar" />
              <span className="audio-playback-bar__bar" />
              <span className="audio-playback-bar__bar" />
            </span>
            <span className="audio-playback-bar__label">Playing audio…</span>
            <button
              className="audio-playback-bar__stop"
              onClick={stopPlayback}
              aria-label="Stop audio playback"
              title="Stop playback"
            >
              ■
            </button>
          </div>
        )}
        <div className="app__input-row">
          <TextInput onSubmit={handleTextSubmit} disabled={inputDisabled} />
          <VoiceButton
            recording={recording}
            onToggle={toggleRecording}
            onStart={startRecording}
            onStop={stopRecording}
            disabled={inputDisabled}
          />
        </div>
      </footer>
    </div>
  );
}
