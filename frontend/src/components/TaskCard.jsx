/**
 * TaskCard — Displays a surfaced task with expandable score breakdown.
 *
 * Shows:
 * - Intent badge with color coding (using Lucide icons, not emojis)
 * - Task content / title
 * - Status indicator
 * - Composite score from Thompson Sampling surfacing engine
 * - Surfacing reason
 * - Project / entity context (if available)
 * - Shame-free action buttons: Done, Later, Let go
 *
 * Expandable section reveals:
 * - Individual signal scores with progress bars
 * - Signal weights from the surfacing query
 * - Reason text for each signal
 * - Thompson sampling statistics (times surfaced, exploration/exploitation)
 *
 * Anti-shame design: no overdue language, no guilt, no forced engagement.
 */
import { useState, useRef, useCallback } from 'react';
import { useDrag } from '@use-gesture/react';
import './TaskCard.css';
import {
  CheckCircle2,
  Clock,
  XCircle,
  ChevronDown,
  Zap,
  Calendar,
  Bell,
  Lightbulb,
  BookOpen,
  RefreshCw,
  HelpCircle,
  Target,
  TrendingUp,
  BarChart3,
  Activity,
} from 'lucide-react';

/* ── Intent configuration ─────────────────────────────────────────── */
const INTENT_ICONS = {
  TASK: Target,
  EVENT: Calendar,
  REMINDER: Bell,
  IDEA: Lightbulb,
  JOURNAL: BookOpen,
  UPDATE: RefreshCw,
  QUESTION: HelpCircle,
};

const INTENT_COLORS = {
  TASK: '#3b82f6',
  EVENT: '#8b5cf6',
  REMINDER: '#f59e0b',
  IDEA: '#10b981',
  JOURNAL: '#ec4899',
  UPDATE: '#6366f1',
  QUESTION: '#14b8a6',
};

const INTENT_LABELS = {
  TASK: 'Task',
  EVENT: 'Event',
  REMINDER: 'Reminder',
  IDEA: 'Idea',
  JOURNAL: 'Journal',
  UPDATE: 'Update',
  QUESTION: 'Question',
};

/* ── Status configuration ─────────────────────────────────────────── */
const STATUS_CONFIG = {
  active: { label: 'Active', className: 'task-card__status--active' },
  deferred: { label: 'Deferred', className: 'task-card__status--deferred' },
  completed: { label: 'Completed', className: 'task-card__status--completed' },
  dropped: { label: 'Dropped', className: 'task-card__status--dropped' },
};

/* ── Signal display names ─────────────────────────────────────────── */
const SIGNAL_LABELS = {
  time_relevance: 'Time Relevance',
  energy_match: 'Energy Match',
  context_relevance: 'Context Match',
  emotional_alignment: 'Emotional Fit',
  momentum: 'Momentum',
  freshness: 'Freshness',
  mood: 'Mood Fit',
  energy: 'Energy Level',
  time_of_day: 'Time of Day',
  calendar: 'Calendar Fit',
  entity_relevance: 'Entity Match',
  behavioral: 'Behavioral',
};

const SIGNAL_ICONS = {
  time_relevance: Clock,
  energy_match: Zap,
  context_relevance: Target,
  emotional_alignment: Activity,
  momentum: TrendingUp,
  freshness: RefreshCw,
  mood: Activity,
  energy: Zap,
  time_of_day: Clock,
  calendar: Calendar,
  entity_relevance: Target,
  behavioral: BarChart3,
};

/**
 * Score bar — visual representation of a 0–1 score value.
 */
function ScoreBar({ value, color = 'var(--accent)' }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div className="task-card__score-bar" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
      <div
        className="task-card__score-bar-fill"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
      <span className="task-card__score-bar-label">{pct}%</span>
    </div>
  );
}

/**
 * Single signal row in the score breakdown.
 */
function SignalRow({ signal, value, reason, weight }) {
  const label = SIGNAL_LABELS[signal] || signal;
  const IconComponent = SIGNAL_ICONS[signal] || BarChart3;

  return (
    <div className="task-card__signal-row">
      <div className="task-card__signal-header">
        <div className="task-card__signal-name">
          <IconComponent size={14} strokeWidth={2} />
          <span>{label}</span>
        </div>
        {weight != null && (
          <span className="task-card__signal-weight" title="Signal weight">
            w: {(weight * 100).toFixed(0)}%
          </span>
        )}
      </div>
      <ScoreBar value={value} />
      {reason && <p className="task-card__signal-reason">{reason}</p>}
    </div>
  );
}

/**
 * Thompson Sampling stats section.
 */
function ThompsonStats({ task, weightsUsed }) {
  const hasWeights = weightsUsed && Object.keys(weightsUsed).length > 0;

  return (
    <div className="task-card__thompson">
      <h4 className="task-card__section-title">
        <BarChart3 size={14} strokeWidth={2} />
        Surfacing Stats
      </h4>
      <div className="task-card__stats-grid">
        <div className="task-card__stat">
          <span className="task-card__stat-value">{task.times_surfaced ?? 0}</span>
          <span className="task-card__stat-label">Times Surfaced</span>
        </div>
        <div className="task-card__stat">
          <span className="task-card__stat-value">
            {task.composite_score != null ? (task.composite_score * 100).toFixed(1) + '%' : '—'}
          </span>
          <span className="task-card__stat-label">Composite Score</span>
        </div>
        <div className="task-card__stat">
          <span className="task-card__stat-value">{task.estimated_energy || '—'}</span>
          <span className="task-card__stat-label">Energy Level</span>
        </div>
        {task.created_at && (
          <div className="task-card__stat">
            <span className="task-card__stat-value">
              {formatRelativeTime(task.created_at)}
            </span>
            <span className="task-card__stat-label">Created</span>
          </div>
        )}
      </div>

      {hasWeights && (
        <div className="task-card__weights">
          <h5 className="task-card__subsection-title">Thompson Sampling Weights</h5>
          <div className="task-card__weights-list">
            {Object.entries(weightsUsed).map(([signal, weight]) => (
              <div className="task-card__weight-item" key={signal}>
                <span className="task-card__weight-name">
                  {SIGNAL_LABELS[signal] || signal}
                </span>
                <div className="task-card__weight-bar-wrap">
                  <div
                    className="task-card__weight-bar"
                    style={{ width: `${Math.min(weight * 100 * 3, 100)}%` }}
                  />
                </div>
                <span className="task-card__weight-value">{(weight * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Format a datetime string as relative time (e.g., "2h ago", "3d ago").
 */
function formatRelativeTime(dateStr) {
  if (!dateStr) return '—';
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

/* ── Main TaskCard Component ──────────────────────────────────────── */

/* ── Swipe gesture thresholds & config ──────────────────────────────── */
const SWIPE_THRESHOLD = 80;  // px to trigger action
const SWIPE_VELOCITY = 0.3;  // velocity threshold for quick swipes
const MAX_TRANSLATE = 140;   // max visual translation in px

const SWIPE_ACTIONS = {
  right: {
    action: 'complete',
    label: 'Done',
    Icon: CheckCircle2,
    color: 'rgba(16, 185, 129, 0.92)',
    bgColor: 'rgba(16, 185, 129, 0.12)',
  },
  left: {
    action: 'drop',
    label: 'Let go',
    Icon: XCircle,
    color: 'rgba(239, 68, 68, 0.92)',
    bgColor: 'rgba(239, 68, 68, 0.12)',
  },
  up: {
    action: 'defer',
    label: 'Later',
    Icon: Clock,
    color: 'rgba(245, 158, 11, 0.92)',
    bgColor: 'rgba(245, 158, 11, 0.12)',
  },
};

/**
 * Determine dominant swipe direction from displacement.
 * Returns 'right', 'left', 'up', or null.
 */
function getSwipeDirection(mx, my) {
  const absX = Math.abs(mx);
  const absY = Math.abs(my);

  // Up swipe: negative Y with enough dominance
  if (my < 0 && absY > absX * 0.8) return 'up';
  // Horizontal dominates
  if (absX > absY * 0.8) return mx > 0 ? 'right' : 'left';
  return null;
}

export function TaskCard({ task, onAction, isNudge, weightsUsed }) {
  const [expanded, setExpanded] = useState(false);
  const detailsRef = useRef(null);

  /* ── Swipe gesture state ─────────────────────────────────────────── */
  const [swipeState, setSwipeState] = useState({
    active: false,
    direction: null,
    x: 0,
    y: 0,
    progress: 0, // 0–1 representing how close to trigger
    triggered: false, // true when past threshold
  });
  const cardRef = useRef(null);

  const bindDrag = useDrag(
    ({ active, movement: [mx, my], velocity: [vx, vy], cancel, memo }) => {
      // Don't allow swipe if no action handler
      if (!onAction) return;

      // Determine initial direction lock on first move
      if (!memo) {
        const dir = getSwipeDirection(mx, my);
        if (!dir) return undefined; // not enough to determine direction yet
        memo = dir;
      }

      const direction = memo;
      const config = SWIPE_ACTIONS[direction];
      if (!config) return memo;

      // Calculate displacement along the swipe axis
      let displacement;
      if (direction === 'up') {
        displacement = Math.max(0, -my); // positive = upward
      } else if (direction === 'right') {
        displacement = Math.max(0, mx);
      } else {
        displacement = Math.max(0, -mx); // left: positive = leftward
      }

      const progress = Math.min(displacement / SWIPE_THRESHOLD, 1);
      const triggered = displacement >= SWIPE_THRESHOLD;

      // Clamp visual translation
      let tx = 0, ty = 0;
      if (direction === 'right') {
        tx = Math.min(mx, MAX_TRANSLATE);
      } else if (direction === 'left') {
        tx = Math.max(mx, -MAX_TRANSLATE);
      } else if (direction === 'up') {
        ty = Math.max(my, -MAX_TRANSLATE);
      }

      if (active) {
        setSwipeState({
          active: true,
          direction,
          x: tx,
          y: ty,
          progress,
          triggered,
        });
      } else {
        // Gesture ended — check if we should fire the action
        const vel = direction === 'up' ? vy : vx;
        const shouldFire = triggered || (displacement > 40 && vel > SWIPE_VELOCITY);

        if (shouldFire && onAction) {
          // Animate out before firing
          setSwipeState({
            active: false,
            direction,
            x: direction === 'right' ? 400 : direction === 'left' ? -400 : 0,
            y: direction === 'up' ? -400 : 0,
            progress: 1,
            triggered: true,
          });
          const taskId = task.task_id || task.id;
          setTimeout(() => {
            onAction(config.action, taskId);
            setSwipeState({ active: false, direction: null, x: 0, y: 0, progress: 0, triggered: false });
          }, 250);
        } else {
          // Snap back with spring
          setSwipeState({ active: false, direction: null, x: 0, y: 0, progress: 0, triggered: false });
        }
      }

      return memo;
    },
    {
      axis: undefined,
      filterTaps: true,
      threshold: 10,
      rubberband: 0.15,
    },
  );

  const toggleExpanded = useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  if (!task) {
    return (
      <div className="task-card task-card--empty">
        <p className="task-card__empty-text">No tasks right now. Enjoy the moment.</p>
      </div>
    );
  }

  const intentKey = task.intent?.toUpperCase() || 'TASK';
  const IntentIcon = INTENT_ICONS[intentKey] || Target;
  const intentColor = INTENT_COLORS[intentKey] || '#3b82f6';
  const intentLabel = INTENT_LABELS[intentKey] || 'Task';
  const statusCfg = STATUS_CONFIG[task.status] || STATUS_CONFIG.active;
  const hasContext = task.project || (task.entity_names && task.entity_names.length > 0);
  const hasSignals = task.signal_scores && task.signal_scores.length > 0;
  const content = task.content || task.title || '';

  // Current swipe config for rendering overlay
  const swipeDir = swipeState.direction;
  const swipeConfig = swipeDir ? SWIPE_ACTIONS[swipeDir] : null;
  const SwipeIcon = swipeConfig?.Icon;

  // Card transform style
  const cardTransform = swipeState.x !== 0 || swipeState.y !== 0
    ? {
        transform: `translate3d(${swipeState.x}px, ${swipeState.y}px, 0)`,
        transition: swipeState.active ? 'none' : 'transform 0.35s cubic-bezier(0.22, 1, 0.36, 1)',
      }
    : {
        transform: 'translate3d(0, 0, 0)',
        transition: 'transform 0.35s cubic-bezier(0.22, 1, 0.36, 1)',
      };

  return (
    <div className="task-card-swipe-container" ref={cardRef}>
      {/* ── Swipe action overlay (behind card) ── */}
      {swipeDir && swipeConfig && (
        <div
          className={`task-card-swipe-overlay task-card-swipe-overlay--${swipeDir}`}
          style={{
            backgroundColor: swipeConfig.bgColor,
            opacity: Math.min(swipeState.progress * 1.5, 1),
          }}
        >
          <div
            className="task-card-swipe-overlay__content"
            style={{
              opacity: swipeState.progress,
              transform: `scale(${0.5 + swipeState.progress * 0.5})`,
            }}
          >
            {SwipeIcon && (
              <SwipeIcon
                size={28}
                strokeWidth={2}
                style={{ color: swipeConfig.color }}
                className={swipeState.triggered ? 'task-card-swipe-icon--triggered' : ''}
              />
            )}
            <span
              className="task-card-swipe-overlay__label"
              style={{ color: swipeConfig.color }}
            >
              {swipeConfig.label}
            </span>
          </div>
        </div>
      )}

      {/* ── Main card with drag binding ── */}
      <div
        {...(onAction ? bindDrag() : {})}
        className={`task-card${isNudge ? ' task-card--nudge' : ''}${expanded ? ' task-card--expanded' : ''}${swipeState.triggered ? ' task-card--swiping-triggered' : swipeState.active ? ' task-card--swiping' : ''}`}
        role="article"
        aria-label={`Task: ${content}. Swipe right to complete, left to drop, up to defer.`}
        style={cardTransform}
        touch-action="pan-y"
      >
      {/* ── Header: intent badge + status + score ── */}
      <div className="task-card__header">
        <span
          className="task-card__intent"
          style={{ backgroundColor: intentColor }}
        >
          <IntentIcon size={14} strokeWidth={2.5} />
          <span>{intentLabel}</span>
        </span>
        <div className="task-card__header-right">
          <span className={`task-card__status ${statusCfg.className}`}>
            {statusCfg.label}
          </span>
          {task.composite_score != null && (
            <span
              className="task-card__score"
              title="Composite surfacing score"
            >
              {Math.round(task.composite_score * 100)}%
            </span>
          )}
        </div>
      </div>

      {/* ── Content ── */}
      <p className="task-card__content">{content}</p>

      {/* ── Surfacing reason ── */}
      {task.surfacing_reason && (
        <p className="task-card__reason">{task.surfacing_reason}</p>
      )}

      {/* ── Context tags ── */}
      {hasContext && (
        <div className="task-card__context">
          {task.project && (
            <span className="task-card__tag task-card__tag--project">{task.project}</span>
          )}
          {task.entity_names &&
            task.entity_names.map((name) => (
              <span className="task-card__tag" key={name}>
                {name}
              </span>
            ))}
        </div>
      )}

      {/* ── Due date ── */}
      {task.due_at && (
        <div className="task-card__due">
          <Clock size={13} strokeWidth={2} />
          <span>Due {formatRelativeTime(task.due_at)}</span>
        </div>
      )}

      {/* ── Action buttons ── */}
      {onAction && (
        <div className="task-card__actions">
          <button
            className="task-card__btn task-card__btn--complete"
            onClick={() => onAction('complete', task.task_id || task.id)}
            aria-label="Complete task"
          >
            <CheckCircle2 size={16} strokeWidth={2} />
            Done
          </button>
          <button
            className="task-card__btn task-card__btn--defer"
            onClick={() => onAction('defer', task.task_id || task.id)}
            aria-label="Defer task"
          >
            <Clock size={16} strokeWidth={2} />
            Later
          </button>
          <button
            className="task-card__btn task-card__btn--drop"
            onClick={() => onAction('drop', task.task_id || task.id)}
            aria-label="Drop task"
          >
            <XCircle size={16} strokeWidth={2} />
            Let go
          </button>
        </div>
      )}

      {/* ── Expandable score breakdown toggle ── */}
      {(hasSignals || weightsUsed) && (
        <button
          className="task-card__expand-toggle"
          onClick={toggleExpanded}
          aria-expanded={expanded}
          aria-controls="task-card-details"
        >
          <span>Score Breakdown</span>
          <span className={`expandable-chevron${expanded ? ' expandable-chevron--open' : ''}`}>
            <ChevronDown size={18} strokeWidth={2} />
          </span>
        </button>
      )}

      {/* ── Expandable details section ── */}
      {(hasSignals || weightsUsed) && (
        <div className={`expandable-section${expanded ? ' expandable-section--open' : ''}`}>
          <div className="expandable-section__inner">
            <div className="expandable-section__content">
              <div
                className="task-card__details"
                id="task-card-details"
                ref={detailsRef}
              >
                {/* Signal scores */}
                {hasSignals && (
                  <div className="task-card__signals">
                    <h4 className="task-card__section-title">
                      <Activity size={14} strokeWidth={2} />
                      Signal Scores
                    </h4>
                    {task.signal_scores.map((s) => (
                      <SignalRow
                        key={s.signal}
                        signal={s.signal}
                        value={s.value}
                        reason={s.reason}
                        weight={weightsUsed?.[s.signal]}
                      />
                    ))}
                  </div>
                )}

                {/* Thompson sampling stats */}
                <ThompsonStats task={task} weightsUsed={weightsUsed} />
              </div>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
