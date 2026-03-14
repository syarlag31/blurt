/**
 * Minimal connection status indicator.
 *
 * Shows a small dot (green=connected, yellow=connecting, red=disconnected)
 * in the header area.
 */
export function ConnectionStatus({ connected, sessionReady }) {
  const status = sessionReady ? 'ready' : connected ? 'connecting' : 'disconnected';
  const label = sessionReady
    ? 'Connected'
    : connected
      ? 'Connecting...'
      : 'Disconnected';

  return (
    <span className={`conn-status conn-status--${status}`} title={label}>
      <span className="conn-status__dot" />
      <span className="conn-status__label">{label}</span>
    </span>
  );
}
