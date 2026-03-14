/**
 * RhythmsView — Daily/weekly rhythm analysis.
 *
 * Layout:
 *   1. Summary Bar — Quick stats chips for detected rhythm types
 *   2. Heatmap Hero — 7x4 grid (days x time slots) with energy visualization
 *   3. Detected Rhythms — Expandable card list sorted by confidence
 *   4. Recommendations — Shame-free, supportive insights
 *
 * Data sources:
 *   GET /api/v1/users/{user_id}/rhythms          — detected rhythms + summary
 *   GET /api/v1/users/{user_id}/rhythms/heatmap   — weekly 28-cell energy grid
 */
import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Activity,
  Zap,
  ZapOff,
  Sun,
  Moon,
  Sunrise,
  Sunset,
  Sparkles,
  TrendingUp,
  TrendingDown,
  Minus,
  Clock,
  BarChart3,
  RefreshCw,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Eye,
  Lightbulb,
  Heart,
  Loader,
} from 'lucide-react';
import { USER_ID, API_BASE } from '../../utils/constants';
import './RhythmsView.css';

/* ── Rhythm type display config ─────────────────────────────── */
const RHYTHM_META = {
  energy_crash:        { label: 'Energy Dip',          icon: ZapOff,      color: '#f87171', colorAlpha: 'rgba(248, 113, 113, 0.12)' },
  energy_peak:         { label: 'Energy Peak',         icon: Zap,         color: '#34d399', colorAlpha: 'rgba(52, 211, 153, 0.12)' },
  creativity_peak:     { label: 'Creative Peak',       icon: Sparkles,    color: '#a78bfa', colorAlpha: 'rgba(167, 139, 250, 0.12)' },
  productivity_window: { label: 'Productivity Window', icon: TrendingUp,  color: '#fbbf24', colorAlpha: 'rgba(251, 191, 36, 0.12)' },
  productivity_dip:    { label: 'Productivity Dip',    icon: TrendingDown,color: '#fb923c', colorAlpha: 'rgba(251, 146, 60, 0.12)' },
  mood_low:            { label: 'Mood Low',            icon: Heart,       color: '#60a5fa', colorAlpha: 'rgba(96, 165, 250, 0.12)' },
  mood_high:           { label: 'Mood High',           icon: Heart,       color: '#f472b6', colorAlpha: 'rgba(244, 114, 182, 0.12)' },
};

const FALLBACK_META = { label: 'Pattern', icon: Activity, color: '#94a3b8', colorAlpha: 'rgba(148, 163, 184, 0.12)' };

/* ── Time/day display helpers ─────────────────────────────── */
const DAY_KEYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'];
const DAY_SHORT = { monday: 'Mon', tuesday: 'Tue', wednesday: 'Wed', thursday: 'Thu', friday: 'Fri', saturday: 'Sat', sunday: 'Sun' };

const TIME_SLOTS = [
  { key: 'morning',   label: 'AM',   fullLabel: '6am–12pm', Icon: Sunrise },
  { key: 'afternoon', label: 'PM',   fullLabel: '12pm–5pm', Icon: Sun },
  { key: 'evening',   label: 'Eve',  fullLabel: '5pm–9pm',  Icon: Sunset },
  { key: 'night',     label: 'Nite', fullLabel: '9pm–6am',  Icon: Moon },
];

function capitalize(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
}

/* ── Heatmap color interpolation ──────────────────────────── */
function energyToColor(score, hasRhythms) {
  if (score <= 0) return 'var(--rhythms-cell-empty)';
  const clamped = Math.min(Math.max(score, 0), 1);
  // Blue → Amber → Green gradient
  if (clamped < 0.35) {
    const a = 0.15 + clamped * 0.6;
    return `rgba(99, 144, 255, ${a.toFixed(2)})`;
  }
  if (clamped < 0.65) {
    const a = 0.25 + (clamped - 0.35) * 1.0;
    return `rgba(251, 191, 36, ${a.toFixed(2)})`;
  }
  const a = 0.35 + (clamped - 0.65) * 1.5;
  return `rgba(16, 185, 129, ${Math.min(a, 0.9).toFixed(2)})`;
}

/* ── Confidence bar ───────────────────────────────────────── */
function ConfidenceBar({ value, color }) {
  const pct = Math.round(value * 100);
  return (
    <div className="rhythm-confidence">
      <div className="rhythm-confidence__track">
        <div
          className="rhythm-confidence__fill"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span className="rhythm-confidence__label">{pct}%</span>
    </div>
  );
}

/* ── Trend icon ───────────────────────────────────────────── */
function TrendIndicator({ trend }) {
  if (trend === 'up') return <TrendingUp size={14} className="rhythm-trend rhythm-trend--up" aria-label="Trending up" />;
  if (trend === 'down') return <TrendingDown size={14} className="rhythm-trend rhythm-trend--down" aria-label="Trending down" />;
  return <Minus size={14} className="rhythm-trend rhythm-trend--stable" aria-label="Stable" />;
}

/* ── Heatmap Cell ─────────────────────────────────────────── */
function HeatmapCell({ dayKey, dayLabel, timeSlot, cell }) {
  const energy = cell?.energy_score || 0;
  const obs = cell?.observation_count || 0;
  const rhythms = cell?.active_rhythms || [];
  const hasRhythms = rhythms.length > 0;

  return (
    <div
      className={`heatmap-cell${hasRhythms ? ' heatmap-cell--rhythm' : ''}${obs === 0 ? ' heatmap-cell--empty' : ''}`}
      style={{ backgroundColor: obs > 0 ? energyToColor(energy, hasRhythms) : undefined }}
      role="gridcell"
      aria-label={`${dayLabel} ${timeSlot.fullLabel}: ${obs} observations, energy ${Math.round(energy * 100)}%${hasRhythms ? ', has rhythms' : ''}`}
      title={`${dayLabel} ${timeSlot.fullLabel}\nEnergy: ${Math.round(energy * 100)}%\nObservations: ${obs}${hasRhythms ? '\nRhythms: ' + rhythms.join(', ') : ''}`}
    >
      {obs > 0 && <span className="heatmap-cell__obs">{obs}</span>}
      {hasRhythms && <span className="heatmap-cell__dot" aria-hidden="true" />}
    </div>
  );
}

/* ── Heatmap Hero ─────────────────────────────────────────── */
function HeatmapHero({ heatmapData }) {
  const cellMap = useMemo(() => {
    if (!heatmapData?.cells) return {};
    const map = {};
    for (const cell of heatmapData.cells) {
      map[`${cell.day_of_week}:${cell.time_of_day}`] = cell;
    }
    return map;
  }, [heatmapData]);

  const totalObs = heatmapData?.total_observations || 0;

  return (
    <section className="rhythms-heatmap" aria-label="Weekly energy heatmap">
      <div className="rhythms-section-header">
        <Activity size={16} aria-hidden="true" />
        <h3 className="rhythms-section-title">Weekly Energy Map</h3>
        {totalObs > 0 && (
          <span className="rhythms-section-badge">{totalObs} obs</span>
        )}
      </div>

      <div className="heatmap-grid" role="grid" aria-label="7-day by 4-slot energy heatmap">
        {/* Corner spacer */}
        <div className="heatmap-corner" role="presentation" />

        {/* Column headers (time slots) */}
        {TIME_SLOTS.map(({ key, label, Icon }) => (
          <div key={key} className="heatmap-col-header" role="columnheader">
            <Icon size={13} aria-hidden="true" />
            <span>{label}</span>
          </div>
        ))}

        {/* Rows (days) */}
        {DAY_KEYS.map((dayKey) => {
          const dayLabel = DAY_SHORT[dayKey];
          return [
            <div key={`row-${dayKey}`} className="heatmap-row-header" role="rowheader">
              {dayLabel}
            </div>,
            ...TIME_SLOTS.map(({ key: timeKey, ...slot }) => (
              <HeatmapCell
                key={`${dayKey}-${timeKey}`}
                dayKey={dayKey}
                dayLabel={dayLabel}
                timeSlot={slot}
                cell={cellMap[`${dayKey}:${timeKey}`]}
              />
            )),
          ];
        })}
      </div>

      {/* Legend */}
      <div className="heatmap-legend" aria-label="Color legend">
        <span className="heatmap-legend__label">Low energy</span>
        <div className="heatmap-legend__gradient" />
        <span className="heatmap-legend__label">High energy</span>
      </div>
    </section>
  );
}

/* ── Single Rhythm Card ───────────────────────────────────── */
function RhythmCard({ rhythm }) {
  const [expanded, setExpanded] = useState(false);
  const meta = RHYTHM_META[rhythm.rhythm_type] || FALLBACK_META;
  const Icon = meta.icon;

  const dayLabel = DAY_SHORT[rhythm.day_of_week] || capitalize(rhythm.day_of_week);
  const timeSlot = TIME_SLOTS.find(s => s.key === rhythm.time_of_day);
  const timeLabel = timeSlot?.fullLabel || capitalize(rhythm.time_of_day);

  return (
    <div className="rhythm-card" role="article">
      <div className="rhythm-card__header">
        <div
          className="rhythm-card__icon-wrap"
          style={{ backgroundColor: meta.colorAlpha, color: meta.color }}
        >
          <Icon size={18} />
        </div>
        <div className="rhythm-card__title-group">
          <h3 className="rhythm-card__label" style={{ color: meta.color }}>
            {meta.label}
          </h3>
          <div className="rhythm-card__when">
            <Clock size={12} />
            <span>{dayLabel} &middot; {timeLabel}</span>
          </div>
        </div>
        <TrendIndicator trend={rhythm.trend} />
      </div>

      <p className="rhythm-card__description">{rhythm.description}</p>

      <div className="rhythm-card__stats">
        <div className="rhythm-card__stat">
          <span className="rhythm-card__stat-label">Confidence</span>
          <ConfidenceBar value={rhythm.confidence} color={meta.color} />
        </div>
        <div className="rhythm-card__stat">
          <span className="rhythm-card__stat-label">Observations</span>
          <span className="rhythm-card__stat-value">
            <BarChart3 size={12} />
            {rhythm.observation_count}
          </span>
        </div>
        {rhythm.is_periodic && (
          <div className="rhythm-card__stat">
            <span className="rhythm-card__stat-label">Recurrence</span>
            <span className="rhythm-card__stat-value rhythm-card__stat-value--accent">
              {Math.round(rhythm.periodicity_strength * 100)}% weekly
            </span>
          </div>
        )}
      </div>

      <button
        className="rhythm-card__expand-btn"
        onClick={() => setExpanded(!expanded)}
        aria-expanded={expanded}
        aria-label={expanded ? 'Show less details' : 'Show more details'}
      >
        <Eye size={14} />
        <span>{expanded ? 'Less' : 'Details'}</span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {expanded && (
        <div className="rhythm-card__details">
          <div className="rhythm-card__detail-row">
            <span className="rhythm-card__detail-label">Z-Score</span>
            <span className="rhythm-card__detail-value">{rhythm.z_score.toFixed(2)}</span>
          </div>
          <div className="rhythm-card__detail-row">
            <span className="rhythm-card__detail-label">Metric Value</span>
            <span className="rhythm-card__detail-value">{rhythm.metric_value.toFixed(3)}</span>
          </div>
          <div className="rhythm-card__detail-row">
            <span className="rhythm-card__detail-label">Population Mean</span>
            <span className="rhythm-card__detail-value">{rhythm.metric_mean.toFixed(3)}</span>
          </div>
          {rhythm.weeks_observed > 0 && (
            <div className="rhythm-card__detail-row">
              <span className="rhythm-card__detail-label">Weeks Observed</span>
              <span className="rhythm-card__detail-value">{rhythm.weeks_observed}</span>
            </div>
          )}
          {rhythm.supporting_evidence?.length > 0 && (
            <div className="rhythm-card__detail-row rhythm-card__detail-row--evidence">
              <span className="rhythm-card__detail-label">Evidence</span>
              <span className="rhythm-card__detail-value">
                {rhythm.supporting_evidence.length} episode{rhythm.supporting_evidence.length !== 1 ? 's' : ''}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Summary Bar ──────────────────────────────────────────── */
function RhythmSummaryBar({ summary }) {
  const items = [
    { label: 'Energy Peaks',  count: summary.energy_peaks,         color: '#34d399' },
    { label: 'Energy Dips',   count: summary.energy_crashes,       color: '#f87171' },
    { label: 'Creative',      count: summary.creativity_peaks,     color: '#a78bfa' },
    { label: 'Productive',    count: summary.productivity_windows, color: '#fbbf24' },
    { label: 'Mood High',     count: summary.mood_highs,           color: '#f472b6' },
    { label: 'Mood Low',      count: summary.mood_lows,            color: '#60a5fa' },
  ].filter(i => i.count > 0);

  if (items.length === 0) return null;

  return (
    <div className="rhythm-summary-bar">
      {items.map(item => (
        <div key={item.label} className="rhythm-summary-chip" style={{ '--chip-color': item.color }}>
          <span className="rhythm-summary-chip__count">{item.count}</span>
          <span className="rhythm-summary-chip__label">{item.label}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Main RhythmsView ─────────────────────────────────────── */
export default function RhythmsView({ refreshKey }) {
  const [rhythmsData, setRhythmsData] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const [rhythmsRes, heatmapRes] = await Promise.all([
        fetch(`${API_BASE}/users/${USER_ID}/rhythms?lookback_weeks=4`),
        fetch(`${API_BASE}/users/${USER_ID}/rhythms/heatmap?lookback_weeks=4`),
      ]);

      if (!rhythmsRes.ok && !heatmapRes.ok) {
        throw new Error(`HTTP ${rhythmsRes.status}`);
      }

      const [rhythms, heatmap] = await Promise.all([
        rhythmsRes.ok ? rhythmsRes.json() : null,
        heatmapRes.ok ? heatmapRes.json() : null,
      ]);

      setRhythmsData(rhythms);
      setHeatmapData(heatmap);
    } catch (err) {
      setError(err.message || 'Failed to load rhythms');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData, refreshKey]);

  /** Sort rhythms by confidence descending */
  const sortedRhythms = useMemo(() => {
    if (!rhythmsData?.rhythms) return [];
    return [...rhythmsData.rhythms].sort((a, b) => b.confidence - a.confidence);
  }, [rhythmsData]);

  const hasHeatmap = heatmapData && (heatmapData.total_observations > 0);
  const hasRhythms = sortedRhythms.length > 0;
  const hasAnyData = hasHeatmap || hasRhythms;

  /* ── Loading state ──────────────────────────────────────── */
  if (loading && !rhythmsData && !heatmapData) {
    return (
      <div className="memory-subview" id="panel-rhythms" role="tabpanel">
        <div className="rhythm-loading">
          <Loader size={24} className="rhythm-loading__spinner" />
          <span className="rhythm-loading__text">Analyzing rhythms...</span>
        </div>
      </div>
    );
  }

  /* ── Error state ────────────────────────────────────────── */
  if (error && !hasAnyData) {
    return (
      <div className="memory-subview" id="panel-rhythms" role="tabpanel">
        <div className="rhythm-error">
          <AlertCircle size={24} />
          <span>Failed to load rhythms: {error}</span>
          <button className="rhythm-error__retry" onClick={fetchData}>
            <RefreshCw size={14} />
            Retry
          </button>
        </div>
      </div>
    );
  }

  /* ── Empty state ────────────────────────────────────────── */
  if (!hasAnyData) {
    return (
      <div className="memory-subview" id="panel-rhythms" role="tabpanel">
        <div className="memory-subview__empty empty-state">
          <Activity size={40} className="empty-state__icon" aria-hidden="true" />
          <h3 className="empty-state__title">No Rhythms Yet</h3>
          <p className="empty-state__description">
            Your daily and weekly patterns will surface here as more data is captured.
          </p>
        </div>
      </div>
    );
  }

  /* ── Populated state ────────────────────────────────────── */
  return (
    <div className="memory-subview rhythms-view" id="panel-rhythms" role="tabpanel">
      {/* Summary chips */}
      {rhythmsData?.summary && <RhythmSummaryBar summary={rhythmsData.summary} />}

      {/* Analysis meta */}
      {rhythmsData && (
        <div className="rhythm-meta">
          <span>{rhythmsData.total_episodes_analyzed} episodes analyzed</span>
          <span className="rhythm-meta__dot">&middot;</span>
          <span>{sortedRhythms.length} pattern{sortedRhythms.length !== 1 ? 's' : ''} found</span>
        </div>
      )}

      {/* Heatmap Hero */}
      {hasHeatmap && <HeatmapHero heatmapData={heatmapData} />}

      {/* Detected Rhythms Section */}
      {hasRhythms && (
        <section className="rhythms-detected" aria-label="Detected rhythms">
          <div className="rhythms-section-header">
            <Zap size={16} aria-hidden="true" />
            <h3 className="rhythms-section-title">Detected Rhythms</h3>
            <span className="rhythms-section-badge">{sortedRhythms.length}</span>
          </div>

          <div className="rhythm-list">
            {sortedRhythms.map((rhythm, idx) => (
              <RhythmCard
                key={`${rhythm.rhythm_type}-${rhythm.day_of_week}-${rhythm.time_of_day}-${idx}`}
                rhythm={rhythm}
              />
            ))}
          </div>
        </section>
      )}

      {/* Recommendations */}
      {rhythmsData?.recommendations?.length > 0 && (
        <section className="rhythm-recommendations" aria-label="Insights">
          <div className="rhythms-section-header">
            <Lightbulb size={16} aria-hidden="true" />
            <h3 className="rhythms-section-title">Gentle Observations</h3>
          </div>
          {rhythmsData.recommendations.map((rec, i) => (
            <p key={i} className="rhythm-recommendations__item">{rec}</p>
          ))}
        </section>
      )}

      {/* Inline refresh indicator */}
      {loading && (
        <div className="rhythm-refresh-indicator" aria-live="polite">
          <Loader size={14} className="rhythm-loading__spinner" />
          <span>Refreshing...</span>
        </div>
      )}
    </div>
  );
}
