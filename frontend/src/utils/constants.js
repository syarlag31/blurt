/**
 * Application-wide constants for Blurt frontend.
 */

// Hardcoded single user — no auth system
export const USER_ID = 'blurt-dogfood-user-001';

// WebSocket endpoint (proxied via Vite in dev)
export const WS_URL =
  import.meta.env.VITE_WS_URL ||
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/audio`;

// API base (proxied via Vite in dev)
export const API_BASE = import.meta.env.VITE_API_BASE || '/api/v1';

// Audio recording config
export const AUDIO_MIME_TYPE = 'audio/webm;codecs=opus';
export const AUDIO_TIMESLICE_MS = 250; // send chunks every 250ms

// Intent labels and colors for metadata cards
export const INTENT_CONFIG = {
  TASK: { label: 'Task', color: '#3b82f6', icon: '✓' },
  EVENT: { label: 'Event', color: '#8b5cf6', icon: '📅' },
  REMINDER: { label: 'Reminder', color: '#f59e0b', icon: '🔔' },
  IDEA: { label: 'Idea', color: '#10b981', icon: '💡' },
  JOURNAL: { label: 'Journal', color: '#ec4899', icon: '📝' },
  UPDATE: { label: 'Update', color: '#6366f1', icon: '🔄' },
  QUESTION: { label: 'Question', color: '#14b8a6', icon: '❓' },
};

// Emotion display helpers
export const EMOTION_COLORS = {
  joy: '#fbbf24',
  trust: '#34d399',
  fear: '#a78bfa',
  surprise: '#f472b6',
  sadness: '#60a5fa',
  disgust: '#fb923c',
  anger: '#ef4444',
  anticipation: '#c084fc',
};
