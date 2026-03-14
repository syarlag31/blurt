/**
 * EventCard — Visual card for detected calendar events.
 *
 * Displayed inline in the chat feed when a message is classified as
 * EVENT intent. Shows extracted event details (title, time, location,
 * attendees) with a sync-to-calendar affordance.
 *
 * Features:
 * - Compact display with key event fields
 * - Sync status indicator (syncing, synced, failed, not connected)
 * - Manual sync/retry button
 * - Connect Google Calendar CTA when not connected
 * - Premium dark theme compatible
 * - 44px+ touch targets
 */
import { useCallback } from 'react';
import {
  Calendar,
  Clock,
  MapPin,
  Users,
  Repeat,
  Check,
  AlertCircle,
  Loader,
  ExternalLink,
} from 'lucide-react';
import { formatEventTime, formatDuration } from '../utils/eventExtractor';
import { SyncState } from '../hooks/useCalendarSync';
import './EventCard.css';

/**
 * @param {object} props
 * @param {object} props.eventData - Extracted event data from eventExtractor
 * @param {string} props.syncState - Current sync state for this event
 * @param {function} props.onSync - Callback to trigger manual sync
 * @param {function} [props.onConnect] - Callback to connect Google Calendar
 * @param {boolean} [props.calendarConnected] - Whether Google Calendar is connected
 */
export function EventCard({
  eventData,
  syncState = SyncState.IDLE,
  onSync,
  onConnect,
  calendarConnected = false,
}) {
  if (!eventData) return null;

  const handleSync = useCallback(
    (e) => {
      e.stopPropagation();
      onSync?.();
    },
    [onSync],
  );

  const handleConnect = useCallback(
    (e) => {
      e.stopPropagation();
      onConnect?.();
    },
    [onConnect],
  );

  const isSynced = syncState === SyncState.SYNCED;
  const isSyncing = syncState === SyncState.SYNCING;
  const isFailed = syncState === SyncState.FAILED;
  const showConnectCta = !calendarConnected && syncState !== SyncState.SYNCED;

  return (
    <div className={`event-card ${isSynced ? 'event-card--synced' : ''}`}>
      {/* Header row: calendar icon + title */}
      <div className="event-card__header">
        <div className="event-card__icon">
          <Calendar size={16} strokeWidth={2} />
        </div>
        <div className="event-card__title-row">
          <span className="event-card__title">{eventData.title}</span>
          {eventData.source === 'heuristic' && (
            <span className="event-card__source-badge">auto-detected</span>
          )}
        </div>
      </div>

      {/* Detail rows */}
      <div className="event-card__details">
        {/* Time */}
        {eventData.startTime && (
          <div className="event-card__detail">
            <Clock size={13} strokeWidth={2} className="event-card__detail-icon" />
            <span className="event-card__detail-text">
              {formatEventTime(eventData.startTime, eventData.allDay)}
              {eventData.durationMinutes && (
                <span className="event-card__duration">
                  {' '}({formatDuration(eventData.durationMinutes)})
                </span>
              )}
            </span>
          </div>
        )}

        {/* Location */}
        {eventData.location && (
          <div className="event-card__detail">
            <MapPin size={13} strokeWidth={2} className="event-card__detail-icon" />
            <span className="event-card__detail-text">{eventData.location}</span>
          </div>
        )}

        {/* Attendees */}
        {eventData.attendees?.length > 0 && (
          <div className="event-card__detail">
            <Users size={13} strokeWidth={2} className="event-card__detail-icon" />
            <span className="event-card__detail-text">
              {eventData.attendees.join(', ')}
            </span>
          </div>
        )}

        {/* Recurrence */}
        {eventData.recurrence && (
          <div className="event-card__detail">
            <Repeat size={13} strokeWidth={2} className="event-card__detail-icon" />
            <span className="event-card__detail-text">{eventData.recurrence}</span>
          </div>
        )}
      </div>

      {/* Sync action row */}
      <div className="event-card__actions">
        {showConnectCta ? (
          <button
            className="event-card__connect-btn"
            onClick={handleConnect}
            type="button"
            aria-label="Connect Google Calendar"
          >
            <ExternalLink size={13} strokeWidth={2} />
            <span>Connect Calendar</span>
          </button>
        ) : (
          <button
            className={`event-card__sync-btn ${isSynced ? 'event-card__sync-btn--synced' : ''} ${isFailed ? 'event-card__sync-btn--failed' : ''}`}
            onClick={handleSync}
            disabled={isSyncing || isSynced}
            type="button"
            aria-label={
              isSynced
                ? 'Synced to Google Calendar'
                : isFailed
                  ? 'Retry sync to Google Calendar'
                  : 'Sync to Google Calendar'
            }
          >
            {isSyncing && <Loader size={13} strokeWidth={2} className="event-card__spin" />}
            {isSynced && <Check size={13} strokeWidth={2.5} />}
            {isFailed && <AlertCircle size={13} strokeWidth={2} />}
            {!isSyncing && !isSynced && !isFailed && (
              <Calendar size={13} strokeWidth={2} />
            )}
            <span>
              {isSyncing
                ? 'Syncing...'
                : isSynced
                  ? 'Synced'
                  : isFailed
                    ? 'Retry'
                    : 'Add to Calendar'}
            </span>
          </button>
        )}

        {/* Confidence indicator */}
        {eventData.confidence != null && (
          <span className="event-card__confidence">
            {Math.round(eventData.confidence * 100)}%
          </span>
        )}
      </div>
    </div>
  );
}
