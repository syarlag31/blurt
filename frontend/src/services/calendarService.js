/**
 * Google Calendar API integration service.
 *
 * Wraps the backend's Google Calendar OAuth2 endpoints and provides
 * methods for authentication management and event operations.
 *
 * Backend endpoints:
 *   GET  /auth/google/connect    → Initiate OAuth2 flow
 *   POST /auth/google/callback   → Handle OAuth2 callback
 *   GET  /auth/google/status     → Check connection status
 *   POST /auth/google/disconnect → Revoke integration
 *
 * Event operations go through the capture/task pipeline:
 *   POST /api/v1/blurt/text      → Create events via blurt capture
 *   POST /api/v1/tasks           → Direct task/event creation
 *   GET  /api/v1/tasks/{id}      → Read task/event details
 *   POST /api/v1/tasks/{id}/feedback → Record feedback (complete/dismiss)
 */

import { USER_ID, API_BASE } from '../utils/constants.js';

// ─── Auth Endpoints ─────────────────────────────────────────────

const AUTH_BASE = '/auth/google';

/**
 * OAuth2 authentication statuses from backend.
 * @enum {string}
 */
export const AuthStatus = {
  NOT_CONFIGURED: 'not_configured',
  AWAITING_AUTH: 'awaiting_auth',
  AUTHENTICATED: 'authenticated',
  TOKEN_EXPIRED: 'token_expired',
  REFRESH_FAILED: 'refresh_failed',
  REVOKED: 'revoked',
};

/**
 * Error class for Calendar service failures.
 */
export class CalendarServiceError extends Error {
  /**
   * @param {string} message
   * @param {number|null} statusCode
   * @param {*} details
   */
  constructor(message, statusCode = null, details = null) {
    super(message);
    this.name = 'CalendarServiceError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

// ─── Internal Helpers ───────────────────────────────────────────

/**
 * Make a fetch request and handle errors uniformly.
 * @param {string} url
 * @param {RequestInit} [options]
 * @returns {Promise<*>}
 */
async function request(url, options = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });

  if (!res.ok) {
    let detail;
    try {
      const body = await res.json();
      detail = body.detail || body.message || JSON.stringify(body);
    } catch {
      detail = res.statusText;
    }
    throw new CalendarServiceError(
      `Calendar API error: ${detail}`,
      res.status,
      detail,
    );
  }

  // Handle 204 No Content
  if (res.status === 204) return null;
  return res.json();
}

/**
 * Build a URL with query parameters.
 * @param {string} base
 * @param {Record<string, string>} params
 * @returns {string}
 */
function buildUrl(base, params = {}) {
  const url = new URL(base, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value != null) url.searchParams.set(key, value);
  }
  return url.toString();
}

// ─── OAuth2 Authentication ──────────────────────────────────────

/**
 * Initiate the Google Calendar OAuth2 connection flow.
 *
 * Returns an authorization URL to redirect the user to Google's
 * consent screen. The backend generates a CSRF state token.
 *
 * @param {string} [userId=USER_ID]
 * @returns {Promise<{ authorization_url: string, state: string }>}
 */
export async function connectCalendar(userId = USER_ID) {
  return request(buildUrl(`${AUTH_BASE}/connect`, { user_id: userId }));
}

/**
 * Handle the OAuth2 callback after user consent.
 *
 * Exchanges the authorization code for tokens. The backend encrypts
 * and stores the tokens. Returns the new connection status.
 *
 * @param {string} code - Authorization code from Google
 * @param {string} [state] - CSRF state token for verification
 * @param {string} [userId=USER_ID]
 * @returns {Promise<{ user_id: string, status: string, connected_email: string|null }>}
 */
export async function handleOAuthCallback(code, state, userId = USER_ID) {
  return request(
    buildUrl(`${AUTH_BASE}/callback`, { user_id: userId, code }),
    { method: 'POST' },
  );
}

/**
 * Check the current Google Calendar connection status.
 *
 * Returns the auth status, connected email, and any errors.
 *
 * @param {string} [userId=USER_ID]
 * @returns {Promise<{ user_id: string, status: string, connected_email: string|null, last_error: string|null }>}
 */
export async function getCalendarStatus(userId = USER_ID) {
  return request(buildUrl(`${AUTH_BASE}/status`, { user_id: userId }));
}

/**
 * Disconnect (revoke) the Google Calendar integration.
 *
 * Revokes the OAuth token at Google and deletes the local
 * encrypted token file.
 *
 * @param {string} [userId=USER_ID]
 * @returns {Promise<{ success: boolean, message: string }>}
 */
export async function disconnectCalendar(userId = USER_ID) {
  return request(
    buildUrl(`${AUTH_BASE}/disconnect`, { user_id: userId }),
    { method: 'POST' },
  );
}

/**
 * Check whether the user is currently authenticated with Google Calendar.
 *
 * @param {string} [userId=USER_ID]
 * @returns {Promise<boolean>}
 */
export async function isCalendarConnected(userId = USER_ID) {
  try {
    const status = await getCalendarStatus(userId);
    return status.status === AuthStatus.AUTHENTICATED;
  } catch {
    return false;
  }
}

// ─── Event Operations ───────────────────────────────────────────
// Events are created/managed through the Blurt capture and task
// pipelines. The backend's Google Calendar client handles the
// bidirectional sync automatically when events are captured.

/**
 * Create a calendar event by submitting it as a blurt.
 *
 * The backend's classification pipeline detects EVENT intent and
 * routes it through the Google Calendar sync pipeline.
 *
 * @param {Object} eventData
 * @param {string} eventData.title - Event title/summary
 * @param {string} [eventData.start_time] - ISO 8601 start time
 * @param {string} [eventData.end_time] - ISO 8601 end time
 * @param {string} [eventData.description] - Event description/notes
 * @param {string} [eventData.location] - Event location
 * @param {string} [userId=USER_ID]
 * @returns {Promise<*>} The created blurt/task response
 */
export async function createCalendarEvent(eventData, userId = USER_ID) {
  // Build a natural language blurt that the classifier will detect as EVENT
  const parts = [`Event: ${eventData.title}`];
  if (eventData.start_time) parts.push(`at ${eventData.start_time}`);
  if (eventData.end_time) parts.push(`until ${eventData.end_time}`);
  if (eventData.location) parts.push(`at ${eventData.location}`);
  if (eventData.description) parts.push(`— ${eventData.description}`);

  return request(`${API_BASE}/blurt/text`, {
    method: 'POST',
    body: JSON.stringify({
      text: parts.join(' '),
      user_id: userId,
    }),
  });
}

/**
 * Create a task/event directly via the tasks API.
 *
 * This bypasses the blurt capture pipeline and creates a task
 * directly in the surfacing store, which will sync to Google
 * Calendar if the integration is connected.
 *
 * @param {Object} taskData
 * @param {string} taskData.title - Task/event title
 * @param {string} [taskData.description] - Description
 * @param {string} [taskData.due_date] - ISO 8601 due date
 * @param {string} [taskData.start_time] - ISO 8601 start time
 * @param {string} [taskData.end_time] - ISO 8601 end time
 * @param {string} [taskData.priority] - Priority level
 * @param {string} [userId=USER_ID]
 * @returns {Promise<*>} The created task response
 */
export async function createTaskEvent(taskData, userId = USER_ID) {
  return request(`${API_BASE}/tasks`, {
    method: 'POST',
    body: JSON.stringify({
      ...taskData,
      user_id: userId,
    }),
  });
}

/**
 * Get a specific task/event by ID.
 *
 * @param {string} taskId - The task ID
 * @returns {Promise<*>} The task details
 */
export async function getTaskEvent(taskId) {
  return request(`${API_BASE}/tasks/${encodeURIComponent(taskId)}`);
}

/**
 * Record feedback on a task/event (accept, dismiss, snooze, complete).
 *
 * For calendar events, "complete" or "dismiss" can trigger
 * cancellation in Google Calendar via the sync pipeline.
 *
 * @param {string} taskId - The task ID
 * @param {Object} feedback
 * @param {'accept'|'dismiss'|'snooze'|'complete'} feedback.action
 * @param {string} [feedback.reason] - Optional reason for the action
 * @param {string} [userId=USER_ID]
 * @returns {Promise<*>} The feedback response
 */
export async function recordEventFeedback(taskId, feedback, userId = USER_ID) {
  return request(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/feedback`, {
    method: 'POST',
    body: JSON.stringify({
      ...feedback,
      user_id: userId,
    }),
  });
}

/**
 * Delete/cancel a calendar event by marking it as dismissed.
 *
 * This records a "dismiss" feedback action which triggers the
 * backend's Google Calendar sync to cancel the event.
 *
 * @param {string} taskId - The task/event ID to cancel
 * @param {string} [reason] - Optional cancellation reason
 * @param {string} [userId=USER_ID]
 * @returns {Promise<*>}
 */
export async function deleteCalendarEvent(taskId, reason, userId = USER_ID) {
  return recordEventFeedback(
    taskId,
    { action: 'dismiss', reason: reason || 'Cancelled from UI' },
    userId,
  );
}

/**
 * Surface tasks/events for the current user with optional context.
 *
 * This is the primary way to list upcoming events — the backend's
 * 21-factor scoring engine considers calendar availability.
 *
 * @param {Object} [context={}] - Optional context for surfacing
 * @param {string} [userId=USER_ID]
 * @returns {Promise<*>} Surfaced tasks including calendar events
 */
export async function surfaceEvents(context = {}, userId = USER_ID) {
  return request(`${API_BASE}/tasks/surface`, {
    method: 'POST',
    body: JSON.stringify({
      user_id: userId,
      ...context,
    }),
  });
}

// ─── Convenience: Full OAuth2 Flow Helper ───────────────────────

/**
 * Initiate the full OAuth2 connection flow.
 *
 * Opens the Google consent screen in a new window/tab. The caller
 * is responsible for listening for the callback (e.g., via
 * postMessage or polling the status endpoint).
 *
 * @param {string} [userId=USER_ID]
 * @returns {Promise<{ window: Window|null, state: string }>}
 */
export async function startOAuthFlow(userId = USER_ID) {
  const { authorization_url, state } = await connectCalendar(userId);

  // Open Google consent screen in a popup
  const popup = window.open(
    authorization_url,
    'google-calendar-auth',
    'width=500,height=700,scrollbars=yes',
  );

  return { window: popup, state };
}

/**
 * Poll for OAuth completion after user authorizes in popup.
 *
 * Checks the status endpoint at intervals until authenticated
 * or the timeout is reached.
 *
 * @param {string} [userId=USER_ID]
 * @param {Object} [options]
 * @param {number} [options.interval=2000] - Poll interval in ms
 * @param {number} [options.timeout=120000] - Timeout in ms (2 min)
 * @param {AbortSignal} [options.signal] - Abort signal
 * @returns {Promise<{ status: string, connected_email: string|null }>}
 */
export async function pollForAuth(userId = USER_ID, options = {}) {
  const { interval = 2000, timeout = 120000, signal } = options;
  const deadline = Date.now() + timeout;

  while (Date.now() < deadline) {
    if (signal?.aborted) {
      throw new CalendarServiceError('Auth polling cancelled');
    }

    const status = await getCalendarStatus(userId);
    if (status.status === AuthStatus.AUTHENTICATED) {
      return status;
    }

    // Wait before next poll
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, interval);
      signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        reject(new CalendarServiceError('Auth polling cancelled'));
      }, { once: true });
    });
  }

  throw new CalendarServiceError('Auth polling timed out', null, { timeout });
}
