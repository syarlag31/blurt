/**
 * Event Intent Detection & Extraction
 *
 * Pure utility functions that parse chat messages and backend metadata
 * for calendar-worthy events. Extracts structured event data:
 * - title, date/time, location, attendees, duration, recurrence
 *
 * Two extraction paths:
 * 1. Backend metadata (preferred) — uses EventMetadata from classification pipeline
 * 2. Client-side heuristic fallback — regex-based extraction from raw text
 *
 * Produces a normalized EventData object suitable for Google Calendar sync.
 */

// ── Regex patterns for client-side fallback extraction ──────────

// Time patterns: "7pm", "7:30 PM", "19:00", "at 3", "3 o'clock"
const TIME_RE =
  /\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|AM|PM)?\b|\b(\d{1,2})\s*o['']?clock\b/g;

// Date patterns: "tomorrow", "next Tuesday", "March 20th", "3/20", "on Saturday"
const RELATIVE_DATE_RE =
  /\b(today|tonight|tomorrow|day after tomorrow)\b/i;

const DAY_OF_WEEK_RE =
  /\b(?:next\s+|this\s+|on\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/i;

const MONTH_DATE_RE =
  /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b/i;

const NUMERIC_DATE_RE =
  /\b(\d{1,2})\/(\d{1,2})(?:\/(\d{2,4}))?\b/;

// Duration patterns: "for 2 hours", "30 minutes", "1hr"
const DURATION_RE =
  /\b(?:for\s+)?(\d+(?:\.\d+)?)\s*(hours?|hrs?|minutes?|mins?|h|m)\b/i;

// Location patterns: "at [Place]", "in [Place]" (after time extraction)
const LOCATION_RE =
  /\bat\s+(?:the\s+)?([A-Z][A-Za-z\s']+?)(?:\s+(?:at|on|from|with|for|in)\b|[.,]|$)/;

// Attendees: "with Sarah", "with the team", "with John and Jane"
const ATTENDEES_RE =
  /\bwith\s+((?:[A-Z][a-z]+)(?:\s+(?:and|&)\s+[A-Z][a-z]+)*(?:\s*,\s*[A-Z][a-z]+)*)/;

// Recurrence: "every Monday", "weekly", "daily", "every week"
const RECURRENCE_RE =
  /\b(every\s+(?:day|week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|daily|weekly|monthly|biweekly)\b/i;

// Month name to index
const MONTH_MAP = {
  january: 0, february: 1, march: 2, april: 3, may: 4, june: 5,
  july: 6, august: 7, september: 8, october: 9, november: 10, december: 11,
};

// Day name to dayOfWeek (0 = Sunday)
const DAY_MAP = {
  sunday: 0, monday: 1, tuesday: 2, wednesday: 3,
  thursday: 4, friday: 5, saturday: 6,
};


// ── Core Detection ──────────────────────────────────────────────

/**
 * Check if a message has EVENT intent from backend classification.
 *
 * @param {object} message - Chat message with optional metadata/ack
 * @returns {boolean}
 */
export function isEventIntent(message) {
  const intent = extractIntentString(message);
  if (!intent) return false;
  return intent.toUpperCase() === 'EVENT';
}

/**
 * Check if a message is syncable to Google Calendar.
 * Requires EVENT intent + sufficient confidence.
 *
 * @param {object} message
 * @param {number} [minConfidence=0.5]
 * @returns {boolean}
 */
export function isSyncableEvent(message, minConfidence = 0.5) {
  if (!isEventIntent(message)) return false;
  const confidence = extractConfidence(message);
  return confidence >= minConfidence;
}


// ── Full Extraction Pipeline ────────────────────────────────────

/**
 * Extract structured event data from a message.
 *
 * Prefers backend-extracted metadata when available, falls back to
 * client-side regex extraction from raw text.
 *
 * @param {object} message - Chat message
 * @returns {EventData|null} Normalized event data, or null if not an event
 */
export function extractEventData(message) {
  const intent = extractIntentString(message);
  if (!intent || intent.toUpperCase() !== 'EVENT') return null;

  const text = message.text || '';
  const metadata = message.metadata || message.ack?.metadata || {};
  const confidence = extractConfidence(message);

  // Path 1: Backend-extracted intent metadata (EventMetadata model)
  const intentMeta = metadata.intent_metadata || metadata.event_metadata || {};
  if (intentMeta.title || intentMeta.start_time) {
    return normalizeBackendEventData(intentMeta, text, confidence);
  }

  // Path 2: Client-side heuristic extraction from raw text
  return extractFromText(text, confidence, metadata.entities);
}

/**
 * Normalize backend EventMetadata into our frontend EventData shape.
 */
function normalizeBackendEventData(meta, rawText, confidence) {
  return {
    title: meta.title || deriveTitle(rawText),
    startTime: meta.start_time ? new Date(meta.start_time) : null,
    endTime: meta.end_time ? new Date(meta.end_time) : null,
    durationMinutes: meta.duration_minutes || null,
    allDay: meta.all_day || false,
    location: meta.location || extractLocation(rawText),
    attendees: meta.attendees || extractAttendees(rawText),
    recurrence: meta.recurrence || extractRecurrence(rawText),
    confidence,
    source: 'backend',
    rawText,
  };
}


// ── Client-Side Heuristic Extraction ────────────────────────────

/**
 * Extract event data from raw text using regex patterns.
 * Best-effort — partial extraction is fine (anti-shame design).
 */
function extractFromText(text, confidence, entities) {
  if (!text) return null;

  const title = deriveTitle(text);
  const timeInfo = extractTimeInfo(text);
  const location = extractLocation(text);
  const attendees = extractAttendees(text);
  const recurrence = extractRecurrence(text);
  const duration = extractDuration(text);

  return {
    title,
    startTime: timeInfo.startTime,
    endTime: timeInfo.endTime,
    durationMinutes: duration,
    allDay: timeInfo.allDay,
    location,
    attendees,
    recurrence,
    confidence,
    source: 'heuristic',
    rawText: text,
  };
}

/**
 * Derive a short event title from raw text.
 * Strips time/date/location expressions to get the core subject.
 */
function deriveTitle(text) {
  if (!text) return 'Untitled Event';

  let title = text
    // Remove common prefixes
    .replace(/^(?:i have |i've got |there's |there is |got )/i, '')
    // Remove time expressions
    .replace(/\b(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)\b/g, '')
    .replace(/\b\d{1,2}\s*o['']?clock\b/g, '')
    // Remove date expressions
    .replace(/\b(?:on\s+)?(?:next\s+)?(?:this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/gi, '')
    .replace(/\b(?:today|tonight|tomorrow|day after tomorrow)\b/gi, '')
    .replace(/\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b/gi, '')
    .replace(/\b\d{1,2}\/\d{1,2}(?:\/\d{2,4})?\b/g, '')
    // Remove relative expressions
    .replace(/\b(?:next week|this week|in the morning|in the evening|in the afternoon)\b/gi, '')
    // Clean up
    .replace(/\s+/g, ' ')
    .trim();

  // Capitalize first letter
  if (title.length > 0) {
    title = title.charAt(0).toUpperCase() + title.slice(1);
  }

  // Truncate if too long
  if (title.length > 60) {
    title = title.substring(0, 57) + '...';
  }

  return title || 'Untitled Event';
}

/**
 * Extract time and date information from text.
 * Returns { startTime: Date|null, endTime: Date|null, allDay: boolean }
 */
function extractTimeInfo(text) {
  const now = new Date();
  let targetDate = null;
  let hours = null;
  let minutes = 0;
  let allDay = false;

  // Extract date
  const relativeMatch = text.match(RELATIVE_DATE_RE);
  if (relativeMatch) {
    const word = relativeMatch[1].toLowerCase();
    targetDate = new Date(now);
    if (word === 'tomorrow') {
      targetDate.setDate(targetDate.getDate() + 1);
    } else if (word === 'day after tomorrow') {
      targetDate.setDate(targetDate.getDate() + 2);
    } else if (word === 'tonight') {
      // tonight means today, default to 8pm if no time specified
      if (!text.match(TIME_RE)) {
        hours = 20;
      }
    }
  }

  const dayMatch = text.match(DAY_OF_WEEK_RE);
  if (dayMatch && !targetDate) {
    const dayName = dayMatch[1].toLowerCase();
    const targetDay = DAY_MAP[dayName];
    targetDate = new Date(now);
    const currentDay = targetDate.getDay();
    let daysUntil = targetDay - currentDay;
    if (daysUntil <= 0) daysUntil += 7; // next occurrence
    targetDate.setDate(targetDate.getDate() + daysUntil);
  }

  const monthMatch = text.match(MONTH_DATE_RE);
  if (monthMatch && !targetDate) {
    const month = MONTH_MAP[monthMatch[1].toLowerCase()];
    const day = parseInt(monthMatch[2], 10);
    targetDate = new Date(now.getFullYear(), month, day);
    // If the date has passed this year, assume next year
    if (targetDate < now) {
      targetDate.setFullYear(targetDate.getFullYear() + 1);
    }
  }

  const numericMatch = text.match(NUMERIC_DATE_RE);
  if (numericMatch && !targetDate) {
    const m = parseInt(numericMatch[1], 10) - 1;
    const d = parseInt(numericMatch[2], 10);
    const y = numericMatch[3]
      ? (numericMatch[3].length === 2 ? 2000 + parseInt(numericMatch[3], 10) : parseInt(numericMatch[3], 10))
      : now.getFullYear();
    targetDate = new Date(y, m, d);
  }

  // Extract time
  // Reset regex lastIndex
  TIME_RE.lastIndex = 0;
  const timeMatch = TIME_RE.exec(text);
  if (timeMatch) {
    if (timeMatch[4]) {
      // "X o'clock" form
      hours = parseInt(timeMatch[4], 10);
    } else {
      hours = parseInt(timeMatch[1], 10);
      minutes = timeMatch[2] ? parseInt(timeMatch[2], 10) : 0;
      const ampm = timeMatch[3]?.toLowerCase();
      if (ampm === 'pm' && hours < 12) hours += 12;
      if (ampm === 'am' && hours === 12) hours = 0;
      // If no am/pm, assume PM for hours < 8 (common for events)
      if (!ampm && hours < 8) hours += 12;
    }
  }

  // If we have a date but no time, it might be an all-day event
  if (targetDate && hours === null) {
    allDay = true;
  }

  // Build startTime
  let startTime = null;
  if (targetDate || hours !== null) {
    startTime = targetDate ? new Date(targetDate) : new Date(now);
    if (hours !== null) {
      startTime.setHours(hours, minutes, 0, 0);
    } else {
      startTime.setHours(0, 0, 0, 0);
    }
  }

  // Extract duration to compute endTime
  const duration = extractDuration(text);
  let endTime = null;
  if (startTime && duration) {
    endTime = new Date(startTime.getTime() + duration * 60000);
  } else if (startTime && !allDay) {
    // Default 1-hour event
    endTime = new Date(startTime.getTime() + 60 * 60000);
  }

  return { startTime, endTime, allDay };
}

/**
 * Extract duration in minutes from text.
 */
function extractDuration(text) {
  const match = text.match(DURATION_RE);
  if (!match) return null;

  const val = parseFloat(match[1]);
  const unit = match[2].toLowerCase();

  if (unit.startsWith('h')) return Math.round(val * 60);
  if (unit.startsWith('m')) return Math.round(val);
  return null;
}

/**
 * Extract location from text.
 */
function extractLocation(text) {
  if (!text) return null;
  const match = text.match(LOCATION_RE);
  return match ? match[1].trim() : null;
}

/**
 * Extract attendees from text.
 */
function extractAttendees(text) {
  if (!text) return [];
  const match = text.match(ATTENDEES_RE);
  if (!match) return [];

  return match[1]
    .split(/\s*(?:,|and|&)\s*/i)
    .map((name) => name.trim())
    .filter((name) => name.length > 0);
}

/**
 * Extract recurrence pattern from text.
 */
function extractRecurrence(text) {
  if (!text) return null;
  const match = text.match(RECURRENCE_RE);
  return match ? match[1].toLowerCase() : null;
}


// ── Helper: extract intent string from various metadata shapes ──

function extractIntentString(message) {
  if (message?.metadata?.intent) return message.metadata.intent;
  if (message?.ack?.metadata?.intent) return message.ack.metadata.intent;
  if (message?.intent) return message.intent;
  return null;
}

function extractConfidence(message) {
  if (message?.metadata?.confidence != null) return message.metadata.confidence;
  if (message?.ack?.metadata?.confidence != null) return message.ack.metadata.confidence;
  if (message?.confidence != null) return message.confidence;
  return 0;
}


// ── Format helpers for display ──────────────────────────────────

/**
 * Format event time for display in EventCard.
 * @param {Date|null} date
 * @param {boolean} allDay
 * @returns {string}
 */
export function formatEventTime(date, allDay) {
  if (!date) return 'Time TBD';
  if (allDay) {
    return date.toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  }
  return date.toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/**
 * Format duration for display.
 * @param {number|null} minutes
 * @returns {string}
 */
export function formatDuration(minutes) {
  if (!minutes) return '';
  if (minutes < 60) return `${minutes}min`;
  const h = Math.floor(minutes / 60);
  const m = minutes % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

/**
 * Build a Google Calendar sync payload from EventData.
 * Matches the backend EventMetadata model shape.
 *
 * @param {object} eventData - Extracted event data
 * @returns {object} Payload for sync API
 */
export function buildCalendarSyncPayload(eventData) {
  return {
    title: eventData.title,
    start_time: eventData.startTime?.toISOString() || null,
    end_time: eventData.endTime?.toISOString() || null,
    duration_minutes: eventData.durationMinutes,
    all_day: eventData.allDay,
    location: eventData.location,
    attendees: eventData.attendees,
    recurrence: eventData.recurrence,
  };
}
