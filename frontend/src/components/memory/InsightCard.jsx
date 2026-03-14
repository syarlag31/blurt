/**
 * InsightCard — Dismissable / confirmable pattern insight card.
 *
 * Shows a learned pattern with:
 *   - Pattern type badge (color-coded)
 *   - Description text
 *   - Confidence meter
 *   - Observation count
 *   - Confirm (reinforce) and Dismiss actions
 *   - Smooth exit animation on dismiss
 *
 * Designed for 44px+ touch targets and premium dark mode.
 */
import { useState, useRef, useCallback } from 'react';
import {
  Check,
  X,
  TrendingUp,
  Clock,
  Calendar,
  Zap,
  Brain,
  Repeat,
  ChevronDown,
  Sparkles,
} from 'lucide-react';
import { useSwipeGesture } from '../../hooks/useSwipeGesture';
import './InsightCard.css';

/** Map pattern_type to display label and icon */
const PATTERN_META = {
  ENERGY_RHYTHM:    { label: 'Energy',      icon: Zap,        color: '#f59e0b' },
  MOOD_CYCLE:       { label: 'Mood',        icon: TrendingUp, color: '#ec4899' },
  TIME_OF_DAY:      { label: 'Time',        icon: Clock,      color: '#6390ff' },
  DAY_OF_WEEK:      { label: 'Weekly',      icon: Calendar,   color: '#8b5cf6' },
  COMPLETION_SIGNAL: { label: 'Completion', icon: Check,      color: '#10b981' },
  SKIP_SIGNAL:      { label: 'Skip',        icon: Repeat,     color: '#ef4444' },
  ENTITY_PATTERN:   { label: 'Entity',      icon: Brain,      color: '#06b6d4' },
};

const DEFAULT_META = { label: 'Insight', icon: Sparkles, color: '#6390ff' };

/** Format confidence as percentage. */
function fmtConfidence(val) {
  if (val == null) return '—';
  return `${Math.round(val * 100)}%`;
}

/** Format relative time from ISO string. */
function fmtRelativeTime(isoStr) {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    const diff = Date.now() - d.getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.floor(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

export default function InsightCard({
  pattern,
  onConfirm,
  onDismiss,
}) {
  const [exiting, setExiting] = useState(false);
  const [confirmed, setConfirmed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [busy, setBusy] = useState(false);
  const cardRef = useRef(null);

  const meta = PATTERN_META[pattern.pattern_type] || DEFAULT_META;
  const Icon = meta.icon;

  // Swipe-to-dismiss: left swipe = dismiss, right swipe = confirm
  const handleSwipe = useCallback(
    async (direction) => {
      if (busy) return;
      setBusy(true);
      if (direction === 'left') {
        setExiting(true);
        setTimeout(async () => {
          await onDismiss?.(pattern.id);
          setBusy(false);
        }, 280);
      } else if (direction === 'right') {
        const ok = await onConfirm?.(pattern.id);
        setBusy(false);
        if (ok !== false) {
          setConfirmed(true);
          setTimeout(() => setConfirmed(false), 1800);
        }
      }
    },
    [busy, onDismiss, onConfirm, pattern.id],
  );

  const {
    bind: swipeBind,
    offset: swipeOffset,
    direction: swipeDir,
    progress: swipeProgress,
    isDismissing: swipeDismissing,
    isThresholdMet: swipeReady,
    springStyle,
  } = useSwipeGesture({
    threshold: 70,
    velocityThreshold: 0.4,
    directions: ['left', 'right'],
    dismissDistance: 1.4,
    resistance: 0.3,
    onSwipe: handleSwipe,
    disabled: busy || exiting,
  });

  const handleConfirm = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    const ok = await onConfirm?.(pattern.id);
    setBusy(false);
    if (ok !== false) {
      setConfirmed(true);
      // Auto-collapse after confirmation feedback
      setTimeout(() => setConfirmed(false), 1800);
    }
  }, [busy, onConfirm, pattern.id]);

  const handleDismiss = useCallback(async () => {
    if (busy) return;
    setBusy(true);
    setExiting(true);
    // Wait for exit animation before calling dismiss
    setTimeout(async () => {
      await onDismiss?.(pattern.id);
      setBusy(false);
    }, 280);
  }, [busy, onDismiss, pattern.id]);

  const toggleExpand = useCallback(() => {
    setExpanded((e) => !e);
  }, []);

  const confidence = pattern.confidence ?? 0;
  const evidenceCount = pattern.observation_count ?? pattern.supporting_evidence?.length ?? 0;

  // Swipe direction hint colors
  const swipeHintClass = swipeDir === 'left'
    ? 'insight-card--swiping-dismiss'
    : swipeDir === 'right'
      ? 'insight-card--swiping-confirm'
      : '';

  return (
    <div
      ref={cardRef}
      className={`insight-card ${exiting ? 'insight-card--exiting' : ''} ${confirmed ? 'insight-card--confirmed' : ''} ${swipeHintClass}`}
      role="article"
      aria-label={`${meta.label} insight: ${pattern.description}`}
    >
      {/* Swipe action indicators */}
      {swipeDir && swipeProgress > 0.05 && !swipeDismissing && (
        <div
          className={`insight-card__swipe-hint insight-card__swipe-hint--${swipeDir === 'left' ? 'dismiss' : 'confirm'}`}
          style={{ opacity: swipeProgress * 0.8 + 0.2 }}
          aria-hidden="true"
        >
          {swipeDir === 'left' ? (
            <><X size={20} /> <span>Dismiss</span></>
          ) : (
            <><Check size={20} /> <span>Confirm</span></>
          )}
        </div>
      )}

      {/* Swipeable surface wraps all card content */}
      <div
        {...swipeBind()}
        className="insight-card__swipe-surface"
        style={swipeDismissing || (swipeOffset.x !== 0 || swipeOffset.y !== 0) ? springStyle : undefined}
      >
      {/* Header row: type badge + time */}
      <div className="insight-card__header">
        <span
          className="insight-card__type-badge"
          style={{ '--badge-color': meta.color }}
        >
          <Icon size={13} aria-hidden="true" />
          {meta.label}
        </span>

        <span className="insight-card__time">
          {fmtRelativeTime(pattern.last_confirmed || pattern.updated_at)}
        </span>
      </div>

      {/* Description */}
      <p className="insight-card__description">
        {pattern.description}
      </p>

      {/* Confidence meter + evidence count */}
      <div className="insight-card__meta">
        <div className="insight-card__confidence">
          <span className="insight-card__confidence-label">Confidence</span>
          <div className="insight-card__confidence-bar">
            <div
              className="insight-card__confidence-fill"
              style={{
                width: `${Math.round(confidence * 100)}%`,
                '--bar-color': confidence >= 0.7 ? '#10b981' : confidence >= 0.4 ? '#f59e0b' : '#ef4444',
              }}
            />
          </div>
          <span className="insight-card__confidence-value">
            {fmtConfidence(confidence)}
          </span>
        </div>

        {evidenceCount > 0 && (
          <span className="insight-card__evidence-count">
            {evidenceCount} observation{evidenceCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Expandable details */}
      {(pattern.parameters || pattern.supporting_evidence?.length > 0) && (
        <>
          <button
            className="insight-card__expand-btn"
            onClick={toggleExpand}
            aria-expanded={expanded}
            aria-label={expanded ? 'Collapse details' : 'Expand details'}
          >
            <span>Details</span>
            <span className={`expandable-chevron${expanded ? ' expandable-chevron--open' : ''}`}>
              <ChevronDown size={14} />
            </span>
          </button>

          <div className={`expandable-section${expanded ? ' expandable-section--open' : ''}`}>
            <div className="expandable-section__inner">
              <div className="expandable-section__content">
                <div className="insight-card__details">
                  {pattern.parameters && Object.keys(pattern.parameters).length > 0 && (
                    <div className="insight-card__params">
                      {Object.entries(pattern.parameters).map(([key, val]) => (
                        <div key={key} className="insight-card__param-row">
                          <span className="insight-card__param-key">{key.replace(/_/g, ' ')}</span>
                          <span className="insight-card__param-val">
                            {typeof val === 'object' ? JSON.stringify(val) : String(val)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {pattern.supporting_evidence?.length > 0 && (
                    <div className="insight-card__evidence">
                      <span className="insight-card__evidence-label">Evidence</span>
                      <ul className="insight-card__evidence-list">
                        {pattern.supporting_evidence.slice(0, 5).map((ev, i) => (
                          <li key={i} className="insight-card__evidence-item">{ev}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Action buttons */}
      <div className="insight-card__actions">
        <button
          className="insight-card__action insight-card__action--dismiss"
          onClick={handleDismiss}
          disabled={busy}
          aria-label="Dismiss this insight"
        >
          <X size={16} aria-hidden="true" />
          <span>Dismiss</span>
        </button>

        <button
          className="insight-card__action insight-card__action--confirm"
          onClick={handleConfirm}
          disabled={busy}
          aria-label="Confirm this insight"
        >
          <Check size={16} aria-hidden="true" />
          <span>{confirmed ? 'Confirmed!' : 'Confirm'}</span>
        </button>
      </div>

      {/* Confirmation flash overlay */}
      {confirmed && (
        <div className="insight-card__confirm-flash" aria-hidden="true">
          <Check size={28} />
        </div>
      )}
      </div>{/* end swipe surface */}
    </div>
  );
}
