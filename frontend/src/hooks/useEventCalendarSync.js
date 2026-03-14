/**
 * useEventCalendarSync — Detects EVENT intents from blurt.created payloads,
 * creates a pending calendar event with 5-second undo window, and syncs
 * to Google Calendar via the backend.
 *
 * Flow:
 * 1. blurt.created arrives with intent === 'EVENT'
 * 2. Pending event is created with `status: 'pending'`
 * 3. UndoToast shows for 5 seconds
 * 4. If user taps "Undo" → event cancelled, status → 'cancelled'
 * 5. If 5s elapses → POST to backend, status → 'synced' or 'failed'
 *
 * Returns:
 * - pendingEvent: the current pending event (or null)
 * - cancelPendingEvent: callback to cancel before sync
 * - eventHistory: array of past event sync attempts
 * - handleBlurtCreated: callback to pipe blurt.created payloads through
 */
import { useCallback, useRef, useState } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';
import {
  extractEventData,
  buildCalendarSyncPayload,
} from '../utils/eventExtractor';

const UNDO_WINDOW_MS = 5000;

/**
 * Extract calendar-relevant fields from a blurt.created payload.
 *
 * Uses the rich eventExtractor module for client-side heuristic parsing
 * of date, time, title, location, attendees, and recurrence patterns.
 * Falls back to entity-based extraction from backend classification metadata.
 *
 * @param {object} payload - blurt.created WebSocket message
 * @returns {object} Structured event details for calendar sync
 */
function extractEventDetails(payload) {
  const entities = payload.entities || [];
  const transcript = payload.transcript || payload.text || '';
  const confidence = payload.confidence || 0;

  // Build a pseudo-message shape for the eventExtractor module
  const pseudoMessage = {
    text: transcript,
    metadata: {
      intent: payload.intent || 'EVENT',
      confidence,
      entities,
      intent_metadata: payload.intent_metadata || payload.event_metadata || null,
    },
  };

  // Rich extraction: parses date/time from natural language, derives clean
  // title, extracts location, attendees, recurrence, and duration
  const richEvent = extractEventData(pseudoMessage);

  // Entity-based fallback for fields the heuristic regex may miss
  const dateEntity = entities.find(
    (e) => e.type === 'datetime' || e.type === 'date' || e.type === 'time' || e.label === 'DATE',
  );
  const locationEntity = entities.find(
    (e) => e.type === 'location' || e.type === 'place' || e.label === 'LOCATION',
  );

  // Merge: prefer rich extraction results, backfill from entity data
  const title = richEvent?.title ||
    (transcript.length > 80 ? transcript.slice(0, 77) + '...' : transcript);
  const startTime = richEvent?.startTime || null;
  const endTime = richEvent?.endTime || null;
  const location = richEvent?.location || locationEntity?.value || locationEntity?.text || null;
  const attendees = richEvent?.attendees || [];
  const recurrence = richEvent?.recurrence || null;
  const durationMinutes = richEvent?.durationMinutes || null;
  const allDay = richEvent?.allDay || false;

  return {
    title,
    entities,
    datetime: dateEntity?.value || dateEntity?.text || (startTime ? startTime.toISOString() : null),
    startTime,
    endTime,
    durationMinutes,
    allDay,
    location,
    attendees,
    recurrence,
    transcript,
    blurt_id: payload.blurt_id || null,
    confidence,
    source: richEvent?.source || 'entity',
    // Pre-build the calendar sync payload matching backend EventMetadata model
    calendarPayload: richEvent ? buildCalendarSyncPayload(richEvent) : null,
  };
}

export function useEventCalendarSync() {
  const [pendingEvent, setPendingEvent] = useState(null);
  const [eventHistory, setEventHistory] = useState([]);
  const timerRef = useRef(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  /**
   * Attempt to sync the event to Google Calendar via backend.
   */
  const syncToCalendar = useCallback(async (eventDetails) => {
    try {
      // First check if Google Calendar is connected
      const statusRes = await fetch('/auth/google/status');
      const statusData = statusRes.ok ? await statusRes.json() : null;
      const isConnected = statusData?.connected || statusData?.status === 'connected';

      if (!isConnected) {
        // Calendar not connected — mark as failed with reason
        const failedEvent = {
          ...eventDetails,
          status: 'no_calendar',
          syncedAt: new Date().toISOString(),
          error: 'Google Calendar not connected',
        };
        setPendingEvent(null);
        setEventHistory((prev) => [failedEvent, ...prev].slice(0, 20));
        return;
      }

      // Create the calendar event via tasks endpoint (events are stored as tasks
      // with intent=EVENT which triggers the backend sync orchestrator →
      // SyncOrchestrator.on_blurt_classified → Google Calendar adapter)
      const calPayload = eventDetails.calendarPayload || {};
      const res = await fetch(`${API_BASE}/tasks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: USER_ID,
          content: eventDetails.title || eventDetails.transcript,
          intent: 'EVENT',
          entities: eventDetails.entities,
          due_at: calPayload.start_time || eventDetails.datetime,
          estimated_duration_minutes: eventDetails.durationMinutes || calPayload.duration_minutes || null,
          metadata: {
            calendar_sync: true,
            title: eventDetails.title,
            start_time: calPayload.start_time,
            end_time: calPayload.end_time,
            duration_minutes: calPayload.duration_minutes,
            all_day: eventDetails.allDay,
            location: eventDetails.location,
            attendees: eventDetails.attendees,
            recurrence: eventDetails.recurrence,
            blurt_id: eventDetails.blurt_id,
            extraction_source: eventDetails.source,
          },
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const syncedEvent = {
          ...eventDetails,
          status: 'synced',
          syncedAt: new Date().toISOString(),
          task_id: data.task_id || data.id,
        };
        setPendingEvent(null);
        setEventHistory((prev) => [syncedEvent, ...prev].slice(0, 20));
      } else {
        throw new Error(`Sync failed: ${res.status}`);
      }
    } catch (err) {
      const failedEvent = {
        ...eventDetails,
        status: 'failed',
        syncedAt: new Date().toISOString(),
        error: err.message,
      };
      setPendingEvent(null);
      setEventHistory((prev) => [failedEvent, ...prev].slice(0, 20));
    }
  }, []);

  /**
   * Cancel the pending event before it syncs.
   */
  const cancelPendingEvent = useCallback(() => {
    clearTimer();
    setPendingEvent((current) => {
      if (current) {
        const cancelled = { ...current, status: 'cancelled', cancelledAt: new Date().toISOString() };
        setEventHistory((prev) => [cancelled, ...prev].slice(0, 20));
      }
      return null;
    });
  }, [clearTimer]);

  /**
   * Handle a blurt.created payload — detect EVENT intent and start sync flow.
   * Returns true if an event was detected, false otherwise.
   */
  const handleBlurtCreated = useCallback(
    (payload) => {
      const intent = (payload.intent || '').toUpperCase();
      if (intent !== 'EVENT') return false;

      // Extract event details
      const eventDetails = extractEventDetails(payload);
      const pending = {
        ...eventDetails,
        status: 'pending',
        createdAt: new Date().toISOString(),
        id: `evt-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      };

      // Cancel any existing pending event
      clearTimer();

      setPendingEvent(pending);

      // Start the undo countdown
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        syncToCalendar(pending);
      }, UNDO_WINDOW_MS);

      return true;
    },
    [clearTimer, syncToCalendar],
  );

  return {
    pendingEvent,
    cancelPendingEvent,
    eventHistory,
    handleBlurtCreated,
    UNDO_WINDOW_MS,
  };
}
