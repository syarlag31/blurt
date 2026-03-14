/**
 * ActivityHeatmap — Weekly activity heatmap hero visual.
 *
 * 7-day x 24-hour SVG grid showing activity intensity with color-coded
 * cells. Fetches data from GET /api/v1/users/{user_id}/temporal/heatmap
 * (falls back to GET /api/v1/users/{user_id}/rhythms/heatmap).
 *
 * Features:
 *   - Color-coded cells (blue-tinted scale for dark, indigo for light)
 *   - Smooth hover/tap micro-interactions with tooltip
 *   - Loading skeleton shimmer
 *   - Responsive sizing for mobile
 *   - Graceful empty state with generated sample data
 */
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Activity } from 'lucide-react';
import { API_BASE, USER_ID } from '../../utils/constants';
import './ActivityHeatmap.css';

/* ── Constants ──────────────────────────────────────────────── */

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const HOURS = Array.from({ length: 24 }, (_, i) => i);
const HOUR_LABELS = [0, 3, 6, 9, 12, 15, 18, 21];

const CELL_SIZE = 12;
const CELL_GAP = 2;
const CELL_RADIUS = 2.5;
const LABEL_WIDTH = 32;
const HEADER_HEIGHT = 18;
const GRID_PADDING = 4;

/* Cell size + gap */
const STEP = CELL_SIZE + CELL_GAP;

/* Total SVG dimensions */
const SVG_WIDTH = LABEL_WIDTH + GRID_PADDING + (24 * STEP) - CELL_GAP;
const SVG_HEIGHT = HEADER_HEIGHT + GRID_PADDING + (7 * STEP) - CELL_GAP;

/* ── Color scale (5 levels: 0=empty, 1-4=increasing intensity) ── */

function intensityLevel(value, max) {
  if (!max || value <= 0) return 0;
  const ratio = value / max;
  if (ratio <= 0.0) return 0;
  if (ratio <= 0.25) return 1;
  if (ratio <= 0.50) return 2;
  if (ratio <= 0.75) return 3;
  return 4;
}

function formatHour(h) {
  if (h === 0) return '12a';
  if (h < 12) return `${h}a`;
  if (h === 12) return '12p';
  return `${h - 12}p`;
}

/* ── Component ──────────────────────────────────────────────── */

export default function ActivityHeatmap({ refreshKey }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tooltip, setTooltip] = useState(null);
  const svgRef = useRef(null);
  const tooltipTimeout = useRef(null);

  /* ── Fetch heatmap data ─────────────────────────────────── */
  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Try temporal heatmap first, then rhythms heatmap as fallback
      let res = await fetch(
        `${API_BASE}/users/${encodeURIComponent(USER_ID)}/temporal/heatmap`
      );
      if (!res.ok) {
        res = await fetch(
          `${API_BASE}/users/${encodeURIComponent(USER_ID)}/rhythms/heatmap?lookback_weeks=4`
        );
      }
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
      // Generate empty grid as fallback
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData, refreshKey]);

  /* ── Parse data into 7x24 grid ─────────────────────────── */
  const { grid, maxValue } = useMemo(() => {
    // Initialize 7x24 grid of zeros
    const g = Array.from({ length: 7 }, () => new Array(24).fill(0));
    let max = 0;

    if (!data) return { grid: g, maxValue: 0 };

    // Handle various backend response shapes:
    // Shape 1: { heatmap: { "monday": { "0": count, ... }, ... } }
    // Shape 2: { heatmap: [[...], ...] } (2D array)
    // Shape 3: { slots: [{ day, hour, count }, ...] }
    // Shape 4: { data: [...] } or direct array

    const hm = data.heatmap || data.data || data;

    if (Array.isArray(hm)) {
      // Array of slot objects or 2D array
      if (hm.length > 0 && Array.isArray(hm[0])) {
        // 2D array: hm[day][hour]
        hm.forEach((row, dayIdx) => {
          if (dayIdx < 7) {
            row.forEach((val, hourIdx) => {
              if (hourIdx < 24) {
                const v = typeof val === 'number' ? val : (val?.count ?? val?.value ?? 0);
                g[dayIdx][hourIdx] = v;
                if (v > max) max = v;
              }
            });
          }
        });
      } else {
        // Array of slot objects: { day, hour, count/value }
        const dayMap = { monday: 0, tuesday: 1, wednesday: 2, thursday: 3, friday: 4, saturday: 5, sunday: 6 };
        hm.forEach((slot) => {
          const di = typeof slot.day === 'number' ? slot.day : (dayMap[slot.day?.toLowerCase()] ?? -1);
          const hi = slot.hour ?? slot.time_slot ?? -1;
          const v = slot.count ?? slot.value ?? slot.interactions ?? 1;
          if (di >= 0 && di < 7 && hi >= 0 && hi < 24) {
            g[di][hi] += v;
            if (g[di][hi] > max) max = g[di][hi];
          }
        });
      }
    } else if (typeof hm === 'object' && hm !== null) {
      // Object keyed by day name or day index
      const dayMap = { monday: 0, tuesday: 1, wednesday: 2, thursday: 3, friday: 4, saturday: 5, sunday: 6 };
      Object.entries(hm).forEach(([dayKey, hours]) => {
        const di = dayMap[dayKey.toLowerCase()] ?? parseInt(dayKey, 10);
        if (isNaN(di) || di < 0 || di >= 7) return;
        if (typeof hours === 'object' && hours !== null) {
          Object.entries(hours).forEach(([hourKey, val]) => {
            const hi = parseInt(hourKey, 10);
            const v = typeof val === 'number' ? val : (val?.count ?? val?.value ?? 0);
            if (hi >= 0 && hi < 24) {
              g[di][hi] = v;
              if (v > max) max = v;
            }
          });
        }
      });
    }

    return { grid: g, maxValue: max };
  }, [data]);

  /* ── Tooltip handlers ──────────────────────────────────── */
  const handleCellInteraction = useCallback((dayIdx, hourIdx, value, event) => {
    // Clear any pending hide timeout
    if (tooltipTimeout.current) {
      clearTimeout(tooltipTimeout.current);
      tooltipTimeout.current = null;
    }

    const svgEl = svgRef.current;
    if (!svgEl) return;

    const rect = svgEl.getBoundingClientRect();
    const cellX = LABEL_WIDTH + GRID_PADDING + hourIdx * STEP + CELL_SIZE / 2;
    const cellY = HEADER_HEIGHT + GRID_PADDING + dayIdx * STEP;
    const scaleX = rect.width / SVG_WIDTH;
    const scaleY = rect.height / SVG_HEIGHT;

    setTooltip({
      day: DAYS[dayIdx],
      hour: formatHour(hourIdx),
      value,
      x: rect.left + cellX * scaleX,
      y: rect.top + cellY * scaleY - 4,
    });
  }, []);

  const handleCellLeave = useCallback(() => {
    tooltipTimeout.current = setTimeout(() => setTooltip(null), 200);
  }, []);

  /* ── Total activity for header ─────────────────────────── */
  const totalActivity = useMemo(
    () => grid.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0),
    [grid],
  );

  /* ── Render ─────────────────────────────────────────────── */

  if (loading) {
    return (
      <div className="activity-heatmap activity-heatmap--loading">
        <div className="activity-heatmap__header">
          <div className="activity-heatmap__title-row">
            <Activity size={16} />
            <span className="activity-heatmap__title">Weekly Activity</span>
          </div>
        </div>
        <div className="activity-heatmap__skeleton">
          {Array.from({ length: 7 }).map((_, i) => (
            <div key={i} className="activity-heatmap__skeleton-row" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="activity-heatmap">
      {/* Header */}
      <div className="activity-heatmap__header">
        <div className="activity-heatmap__title-row">
          <Activity size={16} className="activity-heatmap__icon" />
          <span className="activity-heatmap__title">Weekly Activity</span>
        </div>
        <div className="activity-heatmap__meta">
          {error ? (
            <button className="activity-heatmap__retry" onClick={fetchData}>
              Retry
            </button>
          ) : (
            <span className="activity-heatmap__total">
              {totalActivity} interaction{totalActivity !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </div>

      {/* SVG Heatmap Grid */}
      <div className="activity-heatmap__chart">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
          className="activity-heatmap__svg"
          role="img"
          aria-label="Weekly activity heatmap showing interaction counts by day and hour"
        >
          {/* Hour labels (top) */}
          {HOUR_LABELS.map((h) => (
            <text
              key={`h-${h}`}
              x={LABEL_WIDTH + GRID_PADDING + h * STEP + CELL_SIZE / 2}
              y={HEADER_HEIGHT - 4}
              className="activity-heatmap__hour-label"
              textAnchor="middle"
            >
              {formatHour(h)}
            </text>
          ))}

          {/* Day labels (left) + cells */}
          {DAYS.map((day, dayIdx) => (
            <g key={day}>
              {/* Day label */}
              <text
                x={LABEL_WIDTH - 4}
                y={HEADER_HEIGHT + GRID_PADDING + dayIdx * STEP + CELL_SIZE / 2 + 1}
                className="activity-heatmap__day-label"
                textAnchor="end"
                dominantBaseline="central"
              >
                {day}
              </text>

              {/* Hour cells */}
              {HOURS.map((hour) => {
                const value = grid[dayIdx][hour];
                const level = intensityLevel(value, maxValue);
                return (
                  <rect
                    key={`${dayIdx}-${hour}`}
                    x={LABEL_WIDTH + GRID_PADDING + hour * STEP}
                    y={HEADER_HEIGHT + GRID_PADDING + dayIdx * STEP}
                    width={CELL_SIZE}
                    height={CELL_SIZE}
                    rx={CELL_RADIUS}
                    ry={CELL_RADIUS}
                    className={`activity-heatmap__cell activity-heatmap__cell--l${level}`}
                    onPointerEnter={(e) => handleCellInteraction(dayIdx, hour, value, e)}
                    onPointerLeave={handleCellLeave}
                    onTouchStart={(e) => {
                      e.preventDefault();
                      handleCellInteraction(dayIdx, hour, value, e);
                    }}
                  >
                    <title>{`${day} ${formatHour(hour)}: ${value} interaction${value !== 1 ? 's' : ''}`}</title>
                  </rect>
                );
              })}
            </g>
          ))}
        </svg>

        {/* Floating tooltip */}
        {tooltip && (
          <div
            className="activity-heatmap__tooltip"
            style={{
              '--tt-x': `${tooltip.x}px`,
              '--tt-y': `${tooltip.y}px`,
            }}
          >
            <span className="activity-heatmap__tooltip-day">{tooltip.day}</span>
            <span className="activity-heatmap__tooltip-hour">{tooltip.hour}</span>
            <span className="activity-heatmap__tooltip-value">
              {tooltip.value} interaction{tooltip.value !== 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="activity-heatmap__legend">
        <span className="activity-heatmap__legend-label">Less</span>
        {[0, 1, 2, 3, 4].map((level) => (
          <span
            key={level}
            className={`activity-heatmap__legend-cell activity-heatmap__cell--l${level}`}
          />
        ))}
        <span className="activity-heatmap__legend-label">More</span>
      </div>
    </div>
  );
}
