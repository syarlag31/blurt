/**
 * Complete registry of all backend REST API endpoints.
 * Grouped by router for the Debug tab's endpoint picker.
 */

import { API_BASE } from './constants.js';

/**
 * @typedef {{ method: string, path: string, label: string, description: string }} Endpoint
 * @typedef {{ key: string, label: string, prefix: string, endpoints: Endpoint[] }} EndpointGroup
 */

/** @type {EndpointGroup[]} */
export const ENDPOINT_GROUPS = [
  {
    key: 'health',
    label: 'Health',
    prefix: '',
    endpoints: [
      { method: 'GET', path: '/health', label: 'Health Check', description: 'Server health and uptime status' },
    ],
  },
  {
    key: 'capture',
    label: 'Capture',
    prefix: `${API_BASE}/blurt`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/blurt`, label: 'Capture Blurt', description: 'Capture a raw blurt (voice or text)' },
      { method: 'POST', path: `${API_BASE}/blurt/voice`, label: 'Voice Blurt', description: 'Capture a voice blurt with audio metadata' },
      { method: 'POST', path: `${API_BASE}/blurt/text`, label: 'Text Blurt', description: 'Capture a text blurt (edits/corrections)' },
      { method: 'GET', path: `${API_BASE}/blurt/stats`, label: 'Capture Stats', description: 'Capture pipeline statistics' },
    ],
  },
  {
    key: 'classify',
    label: 'Classification',
    prefix: `${API_BASE}/classify`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/classify`, label: 'Classify Text', description: 'Classify text silently' },
      { method: 'POST', path: `${API_BASE}/classify/batch`, label: 'Batch Classify', description: 'Classify multiple texts in one request' },
      { method: 'GET', path: `${API_BASE}/classify/stats`, label: 'Classify Stats', description: 'Classification pipeline statistics' },
      { method: 'GET', path: `${API_BASE}/classify/health`, label: 'Classify Health', description: 'Classification pipeline health check' },
    ],
  },
  {
    key: 'episodes',
    label: 'Episodes',
    prefix: `${API_BASE}/episodes`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/episodes`, label: 'Store Episode', description: 'Store a new episode in episodic memory' },
      { method: 'GET', path: `${API_BASE}/episodes/{episode_id}`, label: 'Get Episode', description: 'Retrieve a single episode by ID' },
      { method: 'GET', path: `${API_BASE}/episodes/user/{user_id}`, label: 'User Episodes', description: 'Query episodes with optional filters' },
      { method: 'GET', path: `${API_BASE}/episodes/session/{session_id}`, label: 'Session Episodes', description: 'Get all episodes in a session' },
      { method: 'GET', path: `${API_BASE}/episodes/entity/{user_id}/{entity}`, label: 'Entity Timeline', description: 'Episodes mentioning a specific entity' },
      { method: 'GET', path: `${API_BASE}/episodes/emotions/{user_id}`, label: 'Emotion Episodes', description: 'Episodes for emotion analysis' },
      { method: 'POST', path: `${API_BASE}/episodes/search/semantic`, label: 'Semantic Search', description: 'Search episodes by semantic similarity' },
      { method: 'POST', path: `${API_BASE}/episodes/observations`, label: 'Store Observation', description: 'Store a raw observation (blurt)' },
      { method: 'POST', path: `${API_BASE}/episodes/compress`, label: 'Compress Episodes', description: 'Compress episodes into a summary' },
      { method: 'GET', path: `${API_BASE}/episodes/summaries/{user_id}`, label: 'Summaries', description: 'Retrieve episode summaries' },
      { method: 'GET', path: `${API_BASE}/episodes/recall/{user_id}`, label: 'Full Recall', description: 'Raw episodes + compressed summaries' },
    ],
  },
  {
    key: 'recall',
    label: 'Recall',
    prefix: `${API_BASE}/recall`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/recall`, label: 'Recall Query', description: 'Execute a personal history recall query' },
      { method: 'GET', path: `${API_BASE}/recall/stats`, label: 'Recall Stats', description: 'Recall engine statistics' },
    ],
  },
  {
    key: 'patterns',
    label: 'Patterns',
    prefix: `${API_BASE}/users/{user_id}/patterns`,
    endpoints: [
      { method: 'GET', path: `${API_BASE}/users/{user_id}/patterns`, label: 'List Patterns', description: 'Query learned user patterns with filters' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/patterns/summary`, label: 'Pattern Summary', description: 'Patterns grouped by type' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/patterns/{pattern_id}`, label: 'Get Pattern', description: 'Get a specific pattern' },
      { method: 'POST', path: `${API_BASE}/users/{user_id}/patterns`, label: 'Create Pattern', description: 'Create a new learned pattern' },
      { method: 'PUT', path: `${API_BASE}/users/{user_id}/patterns/{pattern_id}/reinforce`, label: 'Reinforce', description: 'Reinforce a pattern' },
      { method: 'PUT', path: `${API_BASE}/users/{user_id}/patterns/{pattern_id}/weaken`, label: 'Weaken', description: 'Weaken a pattern' },
      { method: 'DELETE', path: `${API_BASE}/users/{user_id}/patterns/{pattern_id}`, label: 'Deactivate', description: 'Deactivate a pattern' },
    ],
  },
  {
    key: 'question',
    label: 'Question',
    prefix: `${API_BASE}/question`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/question`, label: 'Ask Question', description: 'Ask a question with tier-aware recall' },
      { method: 'GET', path: `${API_BASE}/question/types`, label: 'Query Types', description: 'List available query types for a tier' },
    ],
  },
  {
    key: 'rhythms',
    label: 'Rhythms',
    prefix: `${API_BASE}/users/{user_id}/rhythms`,
    endpoints: [
      { method: 'GET', path: `${API_BASE}/users/{user_id}/rhythms`, label: 'All Rhythms', description: 'Get all detected rhythms' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/rhythms/current`, label: 'Current Rhythms', description: 'Rhythm context for current moment' },
      { method: 'POST', path: `${API_BASE}/users/{user_id}/rhythms/analyze`, label: 'Analyze', description: 'Trigger fresh rhythm analysis' },
      { method: 'POST', path: `${API_BASE}/users/{user_id}/rhythms/sync-graph`, label: 'Sync Graph', description: 'Sync rhythms into knowledge graph' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/rhythms/heatmap`, label: 'Heatmap', description: 'Weekly energy/mood heatmap' },
    ],
  },
  {
    key: 'tasks',
    label: 'Task Surfacing',
    prefix: `${API_BASE}/tasks`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/tasks/surface`, label: 'Surface Tasks', description: 'Query for surfaced tasks with context' },
      { method: 'GET', path: `${API_BASE}/tasks/surface`, label: 'Quick Surface', description: 'Quick surface with minimal context' },
      { method: 'POST', path: `${API_BASE}/tasks`, label: 'Add Task', description: 'Add a task to the surfacing store' },
      { method: 'GET', path: `${API_BASE}/tasks/{task_id}`, label: 'Get Task', description: 'Get a specific task by ID' },
    ],
  },
  {
    key: 'temporal',
    label: 'Temporal Activity',
    prefix: `${API_BASE}/users/{user_id}/temporal`,
    endpoints: [
      { method: 'GET', path: `${API_BASE}/users/{user_id}/temporal/profile`, label: 'Full Profile', description: 'Full temporal activity profile' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/temporal/heatmap`, label: 'Heatmap', description: '28-cell weekly heatmap' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/temporal/hourly`, label: 'Hourly', description: 'Hour-of-day activity patterns' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/temporal/energy`, label: 'Energy', description: 'Energy level patterns across week' },
      { method: 'GET', path: `${API_BASE}/users/{user_id}/temporal/mood`, label: 'Mood', description: 'Mood/emotion patterns across week' },
      { method: 'POST', path: `${API_BASE}/users/{user_id}/temporal/record`, label: 'Record', description: 'Record a user interaction' },
    ],
  },
  {
    key: 'status',
    label: 'Status / Clear State',
    prefix: `${API_BASE}/status`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/status/check`, label: 'Check Status', description: 'Check if tasks need attention' },
      { method: 'GET', path: `${API_BASE}/status/clear`, label: 'Clear State', description: 'Quick clear-state check with affirmation' },
    ],
  },
  {
    key: 'feedback',
    label: 'Task Feedback',
    prefix: `${API_BASE}/tasks`,
    endpoints: [
      { method: 'POST', path: `${API_BASE}/tasks/{task_id}/feedback`, label: 'Record Feedback', description: 'Record accept/dismiss/snooze/complete' },
      { method: 'GET', path: `${API_BASE}/tasks/{task_id}/feedback`, label: 'Feedback Summary', description: 'Get feedback summary for a task' },
      { method: 'GET', path: `${API_BASE}/feedback/recent`, label: 'Recent Feedback', description: 'Recent feedback events for a user' },
    ],
  },
  {
    key: 'google',
    label: 'Google Calendar',
    prefix: '/auth/google',
    endpoints: [
      { method: 'GET', path: '/auth/google/connect', label: 'Connect', description: 'Initiate Google Calendar OAuth2' },
      { method: 'POST', path: '/auth/google/callback', label: 'Callback', description: 'Handle OAuth2 callback' },
      { method: 'GET', path: '/auth/google/status', label: 'Status', description: 'Check Calendar connection status' },
      { method: 'POST', path: '/auth/google/disconnect', label: 'Disconnect', description: 'Revoke Calendar integration' },
    ],
  },
];

/** Total number of endpoints */
export const TOTAL_ENDPOINTS = ENDPOINT_GROUPS.reduce(
  (sum, g) => sum + g.endpoints.length,
  0
);

/** Method color mapping */
export const METHOD_COLORS = {
  GET: '#10b981',
  POST: '#3b82f6',
  PUT: '#f59e0b',
  DELETE: '#ef4444',
};
