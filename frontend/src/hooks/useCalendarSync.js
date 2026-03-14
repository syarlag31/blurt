/**
 * Calendar sync state constants.
 *
 * Shared enum for visual sync status in EventCard and other components.
 * The actual sync logic lives in useEventCalendarSync.js which handles
 * the undo-window flow.
 */

/**
 * Sync states for individual calendar events.
 * @enum {string}
 */
export const SyncState = {
  IDLE: 'idle',
  CHECKING: 'checking',
  SYNCING: 'syncing',
  SYNCED: 'synced',
  FAILED: 'failed',
  NOT_CONNECTED: 'not_connected',
};
