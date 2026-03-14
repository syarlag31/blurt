/**
 * StatsCard — Reusable metric display card with animated count,
 * status indicator, and optional trend arrow.
 *
 * Props:
 *   icon       — Lucide React icon component (e.g. Brain, ListTodo)
 *   label      — String label above/below the value (e.g. "Memories")
 *   value      — Numeric or string value to display
 *   trend      — Optional: { direction: 'up'|'down'|'neutral', label: string }
 *   status     — Optional: 'active'|'inactive'|'warning'|'error'
 *   variant    — Optional: 'success'|'warning'|'danger'|'info' (icon bg color)
 *   loading    — Optional: boolean, shows skeleton state
 *   onClick    — Optional: tap handler
 */
import { useEffect, useRef, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './StatsCard.css';

export function StatsCard({
  icon: Icon,
  label,
  value,
  trend,
  status,
  variant,
  loading = false,
  onClick,
}) {
  const [animating, setAnimating] = useState(false);
  const prevValue = useRef(value);

  // Trigger pop animation when value changes
  useEffect(() => {
    if (prevValue.current !== value && value != null && !loading) {
      setAnimating(true);
      const timer = setTimeout(() => setAnimating(false), 400);
      prevValue.current = value;
      return () => clearTimeout(timer);
    }
    prevValue.current = value;
  }, [value, loading]);

  const trendDirection = trend?.direction || 'neutral';
  const TrendIcon =
    trendDirection === 'up'
      ? TrendingUp
      : trendDirection === 'down'
        ? TrendingDown
        : Minus;

  const iconClassName = [
    'stats-card__icon',
    variant ? `stats-card__icon--${variant}` : '',
  ]
    .filter(Boolean)
    .join(' ');

  const cardClassName = [
    'stats-card',
    loading ? 'stats-card--loading' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const valueClassName = [
    'stats-card__value',
    animating ? 'stats-card__value--animate' : '',
  ]
    .filter(Boolean)
    .join(' ');

  const Component = onClick ? 'button' : 'div';
  const interactiveProps = onClick
    ? { onClick, type: 'button', role: 'button' }
    : {};

  return (
    <Component className={cardClassName} {...interactiveProps}>
      {Icon && (
        <div className={iconClassName}>
          <Icon size={20} strokeWidth={2} />
        </div>
      )}

      <div className="stats-card__content">
        <span className="stats-card__label">{loading ? '\u00A0' : label}</span>
        <div className="stats-card__value-row">
          <span className={valueClassName}>
            {loading ? '\u00A0' : formatValue(value)}
          </span>
          {trend && !loading && (
            <span className={`stats-card__trend stats-card__trend--${trendDirection}`}>
              <span className="stats-card__trend-icon">
                <TrendIcon size={12} strokeWidth={2.5} />
              </span>
              {trend.label}
            </span>
          )}
        </div>
      </div>

      {status && !loading && (
        <div className={`stats-card__status stats-card__status--${status}`} />
      )}
    </Component>
  );
}

/**
 * Format display value — add commas for large numbers,
 * pass strings through as-is.
 */
function formatValue(val) {
  if (val == null) return '—';
  if (typeof val === 'number') {
    return val.toLocaleString();
  }
  return String(val);
}
