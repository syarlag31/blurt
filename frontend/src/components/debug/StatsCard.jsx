/**
 * StatsCard — A single stat tile for the debug dashboard.
 *
 * Displays a label, primary value, optional trend/subtitle,
 * and a status indicator dot (ok / warn / error / loading).
 */
import { useState } from 'react';

const STATUS_COLORS = {
  ok: 'var(--success)',
  warn: 'var(--warning)',
  error: 'var(--danger)',
  loading: 'var(--text-secondary)',
  unknown: 'var(--text-secondary)',
};

export default function StatsCard({
  label,
  value,
  subtitle,
  status = 'unknown',
  icon,
  onTap,
}) {
  const [pressed, setPressed] = useState(false);

  return (
    <button
      className="stats-card"
      data-status={status}
      aria-label={`${label}: ${value}`}
      style={{
        transform: pressed ? 'scale(0.97)' : 'scale(1)',
      }}
      onPointerDown={() => setPressed(true)}
      onPointerUp={() => setPressed(false)}
      onPointerLeave={() => setPressed(false)}
      onClick={onTap}
    >
      {/* Status dot */}
      <span
        className="stats-card__dot"
        style={{ background: STATUS_COLORS[status] || STATUS_COLORS.unknown }}
        aria-hidden="true"
      />

      {/* Icon */}
      {icon && <span className="stats-card__icon">{icon}</span>}

      {/* Value */}
      <span className="stats-card__value">{value ?? '—'}</span>

      {/* Label */}
      <span className="stats-card__label">{label}</span>

      {/* Subtitle / trend */}
      {subtitle && <span className="stats-card__subtitle">{subtitle}</span>}
    </button>
  );
}
