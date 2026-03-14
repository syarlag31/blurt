/**
 * Endpoint Registry — Complete catalog of all Blurt backend REST endpoints.
 *
 * Each entry defines the HTTP method, path, expected parameters (path, query, body),
 * and the field type to render in the dynamic form.
 *
 * Field types:
 *   - text:     Single-line string input
 *   - number:   Numeric input with optional min/max/step
 *   - json:     Multi-line JSON textarea
 *   - toggle:   Boolean switch
 *   - select:   Dropdown from a fixed set of options
 *   - datetime: Date-time picker
 *   - hidden:   Auto-filled (e.g. user_id), shown as read-only chip
 */

import { USER_ID } from '../../utils/constants';

// ── Field builders (DRY helpers) ────────────────────────────────

const userIdPath = {
  name: 'user_id',
  label: 'User ID',
  type: 'hidden',
  default: USER_ID,
  location: 'path',
};

const userIdQuery = {
  name: 'user_id',
  label: 'User ID',
  type: 'hidden',
  default: USER_ID,
  location: 'query',
};

const userIdBody = {
  name: 'user_id',
  label: 'User ID',
  type: 'hidden',
  default: USER_ID,
  location: 'body',
};

const limitQuery = (def = 20) => ({
  name: 'limit',
  label: 'Limit',
  type: 'number',
  default: def,
  min: 1,
  max: 200,
  location: 'query',
});

const offsetQuery = {
  name: 'offset',
  label: 'Offset',
  type: 'number',
  default: 0,
  min: 0,
  location: 'query',
};

// ── Endpoint categories ─────────────────────────────────────────

export const ENDPOINT_CATEGORIES = [
  {
    id: 'capture',
    label: 'Capture',
    endpoints: [
      {
        id: 'blurt-text',
        label: 'Send Text Blurt',
        method: 'POST',
        path: '/api/v1/blurt/text',
        fields: [
          userIdBody,
          { name: 'raw_text', label: 'Text', type: 'text', required: true, placeholder: 'What\'s on your mind?', location: 'body' },
          { name: 'session_id', label: 'Session ID', type: 'text', placeholder: 'optional', location: 'body' },
          { name: 'time_of_day', label: 'Time of Day', type: 'select', options: ['morning', 'afternoon', 'evening', 'night'], location: 'body' },
          { name: 'day_of_week', label: 'Day of Week', type: 'select', options: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'body' },
          { name: 'preceding_episode_id', label: 'Preceding Episode', type: 'text', placeholder: 'UUID', location: 'body' },
          { name: 'active_task_id', label: 'Active Task', type: 'text', placeholder: 'UUID', location: 'body' },
        ],
      },
      {
        id: 'blurt-voice',
        label: 'Send Voice Blurt',
        method: 'POST',
        path: '/api/v1/blurt/voice',
        fields: [
          userIdBody,
          { name: 'raw_text', label: 'Transcription', type: 'text', required: true, placeholder: 'Transcribed text', location: 'body' },
          { name: 'session_id', label: 'Session ID', type: 'text', placeholder: 'optional', location: 'body' },
          { name: 'audio_duration_ms', label: 'Duration (ms)', type: 'number', default: 0, min: 0, location: 'body' },
          { name: 'transcription_confidence', label: 'Confidence', type: 'number', default: 1.0, min: 0, max: 1, step: 0.01, location: 'body' },
          { name: 'time_of_day', label: 'Time of Day', type: 'select', options: ['morning', 'afternoon', 'evening', 'night'], location: 'body' },
          { name: 'day_of_week', label: 'Day of Week', type: 'select', options: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'body' },
        ],
      },
      {
        id: 'blurt-generic',
        label: 'Send Blurt (generic)',
        method: 'POST',
        path: '/api/v1/blurt',
        fields: [
          userIdBody,
          { name: 'raw_text', label: 'Text', type: 'text', required: true, location: 'body' },
          { name: 'modality', label: 'Modality', type: 'select', options: ['text', 'voice'], default: 'text', location: 'body' },
          { name: 'session_id', label: 'Session ID', type: 'text', placeholder: 'optional', location: 'body' },
          { name: 'time_of_day', label: 'Time of Day', type: 'select', options: ['morning', 'afternoon', 'evening', 'night'], location: 'body' },
          { name: 'day_of_week', label: 'Day of Week', type: 'select', options: ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'body' },
        ],
      },
      {
        id: 'capture-stats',
        label: 'Capture Stats',
        method: 'GET',
        path: '/api/v1/blurt/stats',
        fields: [],
      },
    ],
  },
  {
    id: 'classify',
    label: 'Classification',
    endpoints: [
      {
        id: 'classify-text',
        label: 'Classify Text',
        method: 'POST',
        path: '/api/v1/classify',
        fields: [
          { name: 'text', label: 'Text', type: 'text', required: true, placeholder: 'Text to classify', location: 'body' },
          { name: 'metadata', label: 'Metadata', type: 'json', placeholder: '{}', location: 'body' },
        ],
      },
      {
        id: 'classify-batch',
        label: 'Batch Classify',
        method: 'POST',
        path: '/api/v1/classify/batch',
        fields: [
          { name: 'texts', label: 'Texts (JSON array)', type: 'json', required: true, placeholder: '["text 1", "text 2"]', location: 'body' },
        ],
      },
      {
        id: 'classify-stats',
        label: 'Classification Stats',
        method: 'GET',
        path: '/api/v1/classify/stats',
        fields: [],
      },
      {
        id: 'classify-health',
        label: 'Classification Health',
        method: 'GET',
        path: '/api/v1/classify/health',
        fields: [],
      },
    ],
  },
  {
    id: 'episodes',
    label: 'Episodes',
    endpoints: [
      {
        id: 'episodes-create',
        label: 'Create Episode',
        method: 'POST',
        path: '/api/v1/episodes',
        fields: [
          userIdBody,
          { name: 'raw_text', label: 'Raw Text', type: 'text', required: true, location: 'body' },
          { name: 'modality', label: 'Modality', type: 'select', options: ['text', 'voice'], default: 'text', location: 'body' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['TASK', 'EVENT', 'REMINDER', 'IDEA', 'JOURNAL', 'UPDATE', 'QUESTION'], location: 'body' },
          { name: 'intent_confidence', label: 'Intent Confidence', type: 'number', default: 0.9, min: 0, max: 1, step: 0.01, location: 'body' },
          { name: 'emotion', label: 'Emotion', type: 'json', placeholder: '{"primary": "joy", "intensity": 0.7}', location: 'body' },
          { name: 'entities', label: 'Entities', type: 'json', placeholder: '[{"name": "...", "type": "..."}]', location: 'body' },
          { name: 'behavioral_signal', label: 'Behavioral Signal', type: 'json', placeholder: '{}', location: 'body' },
          { name: 'context', label: 'Context', type: 'json', placeholder: '{}', location: 'body' },
        ],
      },
      {
        id: 'episodes-get',
        label: 'Get Episode',
        method: 'GET',
        path: '/api/v1/episodes/{episode_id}',
        fields: [
          { name: 'episode_id', label: 'Episode ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
        ],
      },
      {
        id: 'episodes-user',
        label: 'User Episodes',
        method: 'GET',
        path: '/api/v1/episodes/user/{user_id}',
        fields: [
          userIdPath,
          { name: 'start', label: 'Start', type: 'datetime', location: 'query' },
          { name: 'end', label: 'End', type: 'datetime', location: 'query' },
          { name: 'entity', label: 'Entity', type: 'text', placeholder: 'Filter by entity name', location: 'query' },
          { name: 'emotion', label: 'Emotion', type: 'text', placeholder: 'Filter by emotion', location: 'query' },
          { name: 'min_intensity', label: 'Min Intensity', type: 'number', min: 0, max: 1, step: 0.1, location: 'query' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['', 'TASK', 'EVENT', 'REMINDER', 'IDEA', 'JOURNAL', 'UPDATE', 'QUESTION'], location: 'query' },
          limitQuery(50),
          offsetQuery,
          { name: 'include_compressed', label: 'Include Compressed', type: 'toggle', default: false, location: 'query' },
        ],
      },
      {
        id: 'episodes-session',
        label: 'Session Episodes',
        method: 'GET',
        path: '/api/v1/episodes/session/{session_id}',
        fields: [
          { name: 'session_id', label: 'Session ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
        ],
      },
      {
        id: 'episodes-entity-timeline',
        label: 'Entity Timeline',
        method: 'GET',
        path: '/api/v1/episodes/entity/{user_id}/{entity_name}',
        fields: [
          userIdPath,
          { name: 'entity_name', label: 'Entity Name', type: 'text', required: true, placeholder: 'e.g. "Project X"', location: 'path' },
          limitQuery(20),
        ],
      },
      {
        id: 'episodes-emotions',
        label: 'Emotions Timeline',
        method: 'GET',
        path: '/api/v1/episodes/emotions/{user_id}',
        fields: [
          userIdPath,
          { name: 'start', label: 'Start', type: 'datetime', required: true, location: 'query' },
          { name: 'end', label: 'End', type: 'datetime', required: true, location: 'query' },
        ],
      },
      {
        id: 'episodes-search',
        label: 'Semantic Search',
        method: 'POST',
        path: '/api/v1/episodes/search/semantic',
        fields: [
          userIdBody,
          { name: 'query_embedding', label: 'Query Embedding', type: 'json', required: true, placeholder: '[0.1, 0.2, ...]', location: 'body' },
          { name: 'limit', label: 'Limit', type: 'number', default: 10, min: 1, max: 100, location: 'body' },
          { name: 'min_similarity', label: 'Min Similarity', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05, location: 'body' },
        ],
      },
      {
        id: 'episodes-observe',
        label: 'Create Observation',
        method: 'POST',
        path: '/api/v1/episodes/observations',
        fields: [
          userIdBody,
          { name: 'raw_text', label: 'Raw Text', type: 'text', required: true, location: 'body' },
          { name: 'modality', label: 'Modality', type: 'select', options: ['text', 'voice'], default: 'text', location: 'body' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['TASK', 'EVENT', 'REMINDER', 'IDEA', 'JOURNAL', 'UPDATE', 'QUESTION'], location: 'body' },
          { name: 'intent_confidence', label: 'Confidence', type: 'number', default: 0.9, min: 0, max: 1, step: 0.01, location: 'body' },
          { name: 'source_working_id', label: 'Source Working ID', type: 'text', placeholder: 'UUID', location: 'body' },
        ],
      },
      {
        id: 'episodes-compress',
        label: 'Compress Episodes',
        method: 'POST',
        path: '/api/v1/episodes/compress',
        fields: [
          userIdBody,
          { name: 'episode_ids', label: 'Episode IDs', type: 'json', required: true, placeholder: '["uuid1", "uuid2"]', location: 'body' },
          { name: 'summary_text', label: 'Summary Text', type: 'text', required: true, location: 'body' },
        ],
      },
      {
        id: 'episodes-summaries',
        label: 'Summaries',
        method: 'GET',
        path: '/api/v1/episodes/summaries/{user_id}',
        fields: [
          userIdPath,
          { name: 'start', label: 'Start', type: 'datetime', location: 'query' },
          { name: 'end', label: 'End', type: 'datetime', location: 'query' },
        ],
      },
      {
        id: 'episodes-recall',
        label: 'Recall Episodes',
        method: 'GET',
        path: '/api/v1/episodes/recall/{user_id}',
        fields: [
          userIdPath,
          { name: 'start', label: 'Start', type: 'datetime', location: 'query' },
          { name: 'end', label: 'End', type: 'datetime', location: 'query' },
          { name: 'entity', label: 'Entity', type: 'text', location: 'query' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['', 'TASK', 'EVENT', 'REMINDER', 'IDEA', 'JOURNAL', 'UPDATE', 'QUESTION'], location: 'query' },
          { name: 'include_compressed', label: 'Include Compressed', type: 'toggle', default: false, location: 'query' },
          limitQuery(50),
          offsetQuery,
        ],
      },
    ],
  },
  {
    id: 'tasks',
    label: 'Tasks',
    endpoints: [
      {
        id: 'tasks-add',
        label: 'Add Task',
        method: 'POST',
        path: '/api/v1/tasks',
        fields: [
          { name: 'content', label: 'Content', type: 'text', required: true, placeholder: 'Task description', location: 'body' },
          userIdBody,
          { name: 'intent', label: 'Intent', type: 'select', options: ['TASK', 'EVENT', 'REMINDER', 'IDEA'], default: 'TASK', location: 'body' },
          { name: 'due_at', label: 'Due At', type: 'datetime', location: 'body' },
          { name: 'estimated_energy', label: 'Energy (1-5)', type: 'number', min: 1, max: 5, step: 1, location: 'body' },
          { name: 'estimated_duration_minutes', label: 'Duration (min)', type: 'number', min: 1, max: 480, location: 'body' },
          { name: 'entity_names', label: 'Entity Names', type: 'json', placeholder: '["name1"]', location: 'body' },
          { name: 'project', label: 'Project', type: 'text', placeholder: 'optional project name', location: 'body' },
          { name: 'tags', label: 'Tags', type: 'json', placeholder: '["tag1", "tag2"]', location: 'body' },
          { name: 'metadata', label: 'Metadata', type: 'json', placeholder: '{}', location: 'body' },
        ],
      },
      {
        id: 'tasks-get',
        label: 'Get Task',
        method: 'GET',
        path: '/api/v1/tasks/{task_id}',
        fields: [
          { name: 'task_id', label: 'Task ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
          userIdQuery,
        ],
      },
      {
        id: 'tasks-surface-post',
        label: 'Surface Tasks (POST)',
        method: 'POST',
        path: '/api/v1/tasks/surface',
        fields: [
          userIdBody,
          { name: 'energy', label: 'Energy (1-5)', type: 'number', default: 3, min: 1, max: 5, step: 1, location: 'body' },
          { name: 'mood_valence', label: 'Mood Valence', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'mood_arousal', label: 'Mood Arousal', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'max_results', label: 'Max Results', type: 'number', default: 10, min: 1, max: 50, location: 'body' },
          { name: 'min_score', label: 'Min Score', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05, location: 'body' },
          { name: 'active_entity_names', label: 'Active Entities', type: 'json', placeholder: '["name"]', location: 'body' },
          { name: 'active_project', label: 'Active Project', type: 'text', location: 'body' },
          { name: 'include_intents', label: 'Include Intents', type: 'json', placeholder: '["TASK", "REMINDER"]', location: 'body' },
          { name: 'tags_filter', label: 'Tags Filter', type: 'json', placeholder: '["tag"]', location: 'body' },
          { name: 'weights', label: 'Weights', type: 'json', placeholder: '{}', location: 'body' },
        ],
      },
      {
        id: 'tasks-surface-get',
        label: 'Surface Tasks (GET)',
        method: 'GET',
        path: '/api/v1/tasks/surface',
        fields: [
          userIdQuery,
          { name: 'energy', label: 'Energy (1-5)', type: 'number', default: 3, min: 1, max: 5, step: 1, location: 'query' },
          { name: 'mood_valence', label: 'Mood Valence', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'query' },
          { name: 'max_results', label: 'Max Results', type: 'number', default: 10, min: 1, max: 50, location: 'query' },
          { name: 'min_score', label: 'Min Score', type: 'number', default: 0.0, min: 0, max: 1, step: 0.05, location: 'query' },
        ],
      },
      {
        id: 'tasks-feedback-post',
        label: 'Submit Feedback',
        method: 'POST',
        path: '/api/v1/tasks/{task_id}/feedback',
        fields: [
          { name: 'task_id', label: 'Task ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
          userIdBody,
          { name: 'action', label: 'Action', type: 'select', required: true, options: ['accept', 'dismiss', 'snooze', 'complete'], location: 'body' },
          { name: 'mood_valence', label: 'Mood Valence', type: 'number', min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'energy_level', label: 'Energy Level', type: 'number', min: 1, max: 5, step: 1, location: 'body' },
          { name: 'snooze_minutes', label: 'Snooze (min)', type: 'number', min: 1, max: 1440, location: 'body' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['', 'TASK', 'EVENT', 'REMINDER'], location: 'body' },
          { name: 'metadata', label: 'Metadata', type: 'json', placeholder: '{}', location: 'body' },
        ],
      },
      {
        id: 'tasks-feedback-get',
        label: 'Get Feedback Summary',
        method: 'GET',
        path: '/api/v1/tasks/{task_id}/feedback',
        fields: [
          { name: 'task_id', label: 'Task ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
        ],
      },
      {
        id: 'feedback-recent',
        label: 'Recent Feedback',
        method: 'GET',
        path: '/api/v1/feedback/recent',
        fields: [
          userIdQuery,
          limitQuery(20),
        ],
      },
    ],
  },
  {
    id: 'recall',
    label: 'Recall',
    endpoints: [
      {
        id: 'recall-query',
        label: 'Recall Query',
        method: 'POST',
        path: '/api/v1/recall',
        fields: [
          userIdBody,
          { name: 'query', label: 'Query', type: 'text', required: true, placeholder: 'What do you want to recall?', location: 'body' },
          { name: 'max_results', label: 'Max Results', type: 'number', default: 10, min: 1, max: 50, location: 'body' },
          { name: 'source_filter', label: 'Source Filter', type: 'json', placeholder: '["episodes", "tasks"]', location: 'body' },
          { name: 'time_start', label: 'Time Start', type: 'datetime', location: 'body' },
          { name: 'time_end', label: 'Time End', type: 'datetime', location: 'body' },
        ],
      },
      {
        id: 'recall-stats',
        label: 'Recall Stats',
        method: 'GET',
        path: '/api/v1/recall/stats',
        fields: [],
      },
    ],
  },
  {
    id: 'question',
    label: 'Question',
    endpoints: [
      {
        id: 'question-ask',
        label: 'Ask Question',
        method: 'POST',
        path: '/api/v1/question',
        fields: [
          userIdBody,
          { name: 'query', label: 'Query', type: 'text', required: true, placeholder: 'Ask anything about your data...', location: 'body' },
          { name: 'tier', label: 'Tier', type: 'select', options: ['free', 'basic', 'premium'], default: 'free', location: 'body' },
          { name: 'query_type', label: 'Query Type', type: 'text', placeholder: 'e.g. entity_lookup', location: 'body' },
          { name: 'entity_name', label: 'Entity Name', type: 'text', location: 'body' },
          { name: 'entity_type', label: 'Entity Type', type: 'text', location: 'body' },
          { name: 'max_results', label: 'Max Results', type: 'number', default: 10, min: 1, max: 50, location: 'body' },
          { name: 'time_range_days', label: 'Time Range (days)', type: 'number', default: 30, min: 1, max: 365, location: 'body' },
        ],
      },
      {
        id: 'question-types',
        label: 'Available Query Types',
        method: 'GET',
        path: '/api/v1/question/types',
        fields: [
          { name: 'tier', label: 'Tier', type: 'select', options: ['free', 'basic', 'premium'], default: 'free', location: 'query' },
        ],
      },
    ],
  },
  {
    id: 'status',
    label: 'Status / Clear',
    endpoints: [
      {
        id: 'status-check',
        label: 'Status Check',
        method: 'POST',
        path: '/api/v1/status/check',
        fields: [
          userIdBody,
          { name: 'energy', label: 'Energy (1-5)', type: 'number', default: 3, min: 1, max: 5, step: 1, location: 'body' },
          { name: 'current_valence', label: 'Valence', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'current_arousal', label: 'Arousal', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'time_of_day', label: 'Time of Day', type: 'select', options: ['morning', 'afternoon', 'evening', 'night'], location: 'body' },
        ],
      },
      {
        id: 'status-clear',
        label: 'Get Clear State',
        method: 'GET',
        path: '/api/v1/status/clear',
        fields: [],
      },
    ],
  },
  {
    id: 'patterns',
    label: 'Patterns',
    endpoints: [
      {
        id: 'patterns-list',
        label: 'List Patterns',
        method: 'GET',
        path: '/api/v1/users/{user_id}/patterns',
        fields: [
          userIdPath,
          { name: 'type', label: 'Pattern Type', type: 'text', location: 'query' },
          { name: 'day', label: 'Day', type: 'select', options: ['', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'query' },
          { name: 'time', label: 'Time', type: 'select', options: ['', 'morning', 'afternoon', 'evening', 'night'], location: 'query' },
          { name: 'min_confidence', label: 'Min Confidence', type: 'number', min: 0, max: 1, step: 0.1, location: 'query' },
          { name: 'active', label: 'Active Only', type: 'toggle', default: true, location: 'query' },
          limitQuery(20),
          offsetQuery,
        ],
      },
      {
        id: 'patterns-summary',
        label: 'Patterns Summary',
        method: 'GET',
        path: '/api/v1/users/{user_id}/patterns/summary',
        fields: [userIdPath],
      },
      {
        id: 'patterns-get',
        label: 'Get Pattern',
        method: 'GET',
        path: '/api/v1/users/{user_id}/patterns/{pattern_id}',
        fields: [
          userIdPath,
          { name: 'pattern_id', label: 'Pattern ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
        ],
      },
      {
        id: 'patterns-create',
        label: 'Create Pattern',
        method: 'POST',
        path: '/api/v1/users/{user_id}/patterns',
        fields: [
          userIdPath,
          { name: 'pattern_type', label: 'Pattern Type', type: 'text', required: true, placeholder: 'e.g. temporal, behavioral', location: 'body' },
          { name: 'description', label: 'Description', type: 'text', required: true, location: 'body' },
          { name: 'parameters', label: 'Parameters', type: 'json', placeholder: '{}', location: 'body' },
          { name: 'confidence', label: 'Confidence', type: 'number', default: 0.5, min: 0, max: 1, step: 0.05, location: 'body' },
          { name: 'observation_count', label: 'Observation Count', type: 'number', default: 1, min: 0, location: 'body' },
          { name: 'supporting_evidence', label: 'Evidence', type: 'json', placeholder: '[]', location: 'body' },
        ],
      },
      {
        id: 'patterns-reinforce',
        label: 'Reinforce Pattern',
        method: 'PUT',
        path: '/api/v1/users/{user_id}/patterns/{pattern_id}/reinforce',
        fields: [
          userIdPath,
          { name: 'pattern_id', label: 'Pattern ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
          { name: 'evidence', label: 'Evidence', type: 'text', required: true, location: 'body' },
          { name: 'confidence_boost', label: 'Confidence Boost', type: 'number', default: 0.05, min: 0, max: 0.5, step: 0.01, location: 'body' },
        ],
      },
      {
        id: 'patterns-weaken',
        label: 'Weaken Pattern',
        method: 'PUT',
        path: '/api/v1/users/{user_id}/patterns/{pattern_id}/weaken',
        fields: [
          userIdPath,
          { name: 'pattern_id', label: 'Pattern ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
          { name: 'confidence_penalty', label: 'Confidence Penalty', type: 'number', default: 0.05, min: 0, max: 0.5, step: 0.01, location: 'body' },
        ],
      },
      {
        id: 'patterns-delete',
        label: 'Delete Pattern',
        method: 'DELETE',
        path: '/api/v1/users/{user_id}/patterns/{pattern_id}',
        fields: [
          userIdPath,
          { name: 'pattern_id', label: 'Pattern ID', type: 'text', required: true, placeholder: 'UUID', location: 'path' },
        ],
      },
    ],
  },
  {
    id: 'rhythms',
    label: 'Rhythms',
    endpoints: [
      {
        id: 'rhythms-list',
        label: 'User Rhythms',
        method: 'GET',
        path: '/api/v1/users/{user_id}/rhythms',
        fields: [
          userIdPath,
          { name: 'lookback_weeks', label: 'Lookback (weeks)', type: 'number', default: 4, min: 1, max: 52, location: 'query' },
          { name: 'rhythm_type', label: 'Rhythm Type', type: 'text', location: 'query' },
          { name: 'day', label: 'Day', type: 'select', options: ['', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'query' },
          { name: 'time', label: 'Time', type: 'select', options: ['', 'morning', 'afternoon', 'evening', 'night'], location: 'query' },
          { name: 'min_confidence', label: 'Min Confidence', type: 'number', min: 0, max: 1, step: 0.1, location: 'query' },
        ],
      },
      {
        id: 'rhythms-current',
        label: 'Current Rhythm',
        method: 'GET',
        path: '/api/v1/users/{user_id}/rhythms/current',
        fields: [
          userIdPath,
          { name: 'lookback_weeks', label: 'Lookback (weeks)', type: 'number', default: 4, min: 1, max: 52, location: 'query' },
        ],
      },
      {
        id: 'rhythms-analyze',
        label: 'Analyze Rhythms',
        method: 'POST',
        path: '/api/v1/users/{user_id}/rhythms/analyze',
        fields: [
          userIdPath,
          { name: 'lookback_weeks', label: 'Lookback (weeks)', type: 'number', default: 4, min: 1, max: 52, location: 'body' },
        ],
      },
      {
        id: 'rhythms-sync-graph',
        label: 'Sync to Graph',
        method: 'POST',
        path: '/api/v1/users/{user_id}/rhythms/sync-graph',
        fields: [
          userIdPath,
          { name: 'lookback_weeks', label: 'Lookback (weeks)', type: 'number', default: 4, min: 1, max: 52, location: 'body' },
          { name: 'min_confidence', label: 'Min Confidence', type: 'number', default: 0.3, min: 0, max: 1, step: 0.05, location: 'body' },
        ],
      },
      {
        id: 'rhythms-heatmap',
        label: 'Weekly Heatmap',
        method: 'GET',
        path: '/api/v1/users/{user_id}/rhythms/heatmap',
        fields: [
          userIdPath,
          { name: 'lookback_weeks', label: 'Lookback (weeks)', type: 'number', default: 4, min: 1, max: 52, location: 'query' },
        ],
      },
    ],
  },
  {
    id: 'temporal',
    label: 'Temporal',
    endpoints: [
      {
        id: 'temporal-profile',
        label: 'Temporal Profile',
        method: 'GET',
        path: '/api/v1/users/{user_id}/temporal/profile',
        fields: [userIdPath],
      },
      {
        id: 'temporal-heatmap',
        label: 'Temporal Heatmap',
        method: 'GET',
        path: '/api/v1/users/{user_id}/temporal/heatmap',
        fields: [userIdPath],
      },
      {
        id: 'temporal-hourly',
        label: 'Hourly Breakdown',
        method: 'GET',
        path: '/api/v1/users/{user_id}/temporal/hourly',
        fields: [
          userIdPath,
          { name: 'day', label: 'Day', type: 'select', options: ['', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'], location: 'query' },
        ],
      },
      {
        id: 'temporal-energy',
        label: 'Energy Profile',
        method: 'GET',
        path: '/api/v1/users/{user_id}/temporal/energy',
        fields: [userIdPath],
      },
      {
        id: 'temporal-mood',
        label: 'Mood Profile',
        method: 'GET',
        path: '/api/v1/users/{user_id}/temporal/mood',
        fields: [userIdPath],
      },
      {
        id: 'temporal-record',
        label: 'Record Interaction',
        method: 'POST',
        path: '/api/v1/users/{user_id}/temporal/record',
        fields: [
          userIdPath,
          { name: 'energy_level', label: 'Energy (1-5)', type: 'number', default: 3, min: 1, max: 5, step: 1, location: 'body' },
          { name: 'valence', label: 'Valence', type: 'number', default: 0.5, min: -1, max: 1, step: 0.1, location: 'body' },
          { name: 'primary_emotion', label: 'Emotion', type: 'select', options: ['', 'joy', 'trust', 'fear', 'surprise', 'sadness', 'disgust', 'anger', 'anticipation'], location: 'body' },
          { name: 'emotion_intensity', label: 'Intensity', type: 'number', default: 0.5, min: 0, max: 1, step: 0.1, location: 'body' },
          { name: 'intent', label: 'Intent', type: 'select', options: ['', 'TASK', 'EVENT', 'REMINDER', 'IDEA', 'JOURNAL', 'UPDATE', 'QUESTION'], location: 'body' },
          { name: 'word_count', label: 'Word Count', type: 'number', min: 0, location: 'body' },
          { name: 'modality', label: 'Modality', type: 'select', options: ['text', 'voice'], default: 'text', location: 'body' },
          { name: 'task_created', label: 'Task Created', type: 'toggle', default: false, location: 'body' },
          { name: 'task_completed', label: 'Task Completed', type: 'toggle', default: false, location: 'body' },
          { name: 'task_skipped', label: 'Task Skipped', type: 'toggle', default: false, location: 'body' },
          { name: 'task_dismissed', label: 'Task Dismissed', type: 'toggle', default: false, location: 'body' },
        ],
      },
    ],
  },
  {
    id: 'auth',
    label: 'Google Calendar',
    endpoints: [
      {
        id: 'auth-connect',
        label: 'Connect Google',
        method: 'GET',
        path: '/auth/google/connect',
        fields: [userIdQuery],
      },
      {
        id: 'auth-callback',
        label: 'Auth Callback',
        method: 'POST',
        path: '/auth/google/callback',
        fields: [
          userIdQuery,
          { name: 'code', label: 'Auth Code', type: 'text', required: true, placeholder: 'OAuth code', location: 'query' },
        ],
      },
      {
        id: 'auth-status',
        label: 'Auth Status',
        method: 'GET',
        path: '/auth/google/status',
        fields: [userIdQuery],
      },
      {
        id: 'auth-disconnect',
        label: 'Disconnect Google',
        method: 'POST',
        path: '/auth/google/disconnect',
        fields: [userIdQuery],
      },
    ],
  },
  {
    id: 'health',
    label: 'Health',
    endpoints: [
      {
        id: 'health-check',
        label: 'Health Check',
        method: 'GET',
        path: '/health',
        fields: [],
      },
    ],
  },
];

/**
 * Flat lookup: endpointId -> endpoint definition
 */
export const ENDPOINTS_BY_ID = {};
for (const cat of ENDPOINT_CATEGORIES) {
  for (const ep of cat.endpoints) {
    ENDPOINTS_BY_ID[ep.id] = { ...ep, category: cat.id, categoryLabel: cat.label };
  }
}

/**
 * Method color map for HTTP verb badges.
 */
export const METHOD_COLORS = {
  GET: 'var(--success)',
  POST: 'var(--accent)',
  PUT: 'var(--warning)',
  DELETE: 'var(--danger)',
  PATCH: 'var(--info, var(--accent-secondary))',
};
