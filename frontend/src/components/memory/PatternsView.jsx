/**
 * PatternsView — Recurring themes and patterns in memory.
 *
 * Fetches learned patterns from the backend and renders them
 * as dismissable/confirmable InsightCards. Groups patterns by
 * type and shows a summary header.
 *
 * Supports:
 *   - Confirm (reinforce) — strengthens pattern confidence
 *   - Dismiss (deactivate) — removes pattern with exit animation
 *   - Expandable details per card (parameters, evidence)
 *   - Auto-refresh on blurt.created via refreshKey prop
 *   - Empty state when no patterns are detected yet
 *   - Loading + error states
 *   - Pattern type filter chips
 */
import { useState, useMemo, useCallback } from 'react';
import { Sparkles, Loader, AlertCircle, Filter } from 'lucide-react';
import { usePatterns } from '../../hooks/usePatterns';
import InsightCard from './InsightCard';
import './PatternsView.css';

/** Available pattern type filters. */
const PATTERN_TYPES = [
  { id: '', label: 'All' },
  { id: 'ENERGY_RHYTHM', label: 'Energy' },
  { id: 'MOOD_CYCLE', label: 'Mood' },
  { id: 'TIME_OF_DAY', label: 'Time' },
  { id: 'DAY_OF_WEEK', label: 'Weekly' },
  { id: 'COMPLETION_SIGNAL', label: 'Completion' },
  { id: 'SKIP_SIGNAL', label: 'Skip' },
  { id: 'ENTITY_PATTERN', label: 'Entity' },
];

export default function PatternsView({ refreshKey }) {
  const [typeFilter, setTypeFilter] = useState('');

  const filterOpts = useMemo(
    () => (typeFilter ? { type: typeFilter } : {}),
    [typeFilter]
  );

  const {
    patterns,
    loading,
    error,
    reinforcePattern,
    dismissPattern,
  } = usePatterns(refreshKey, filterOpts);

  const handleTypeFilter = useCallback((id) => {
    setTypeFilter((prev) => (prev === id ? '' : id));
  }, []);

  // Group patterns by confidence tier for display ordering
  const sortedPatterns = useMemo(() => {
    return [...patterns].sort((a, b) => {
      // Higher confidence first, then most recently confirmed
      const confDiff = (b.confidence ?? 0) - (a.confidence ?? 0);
      if (Math.abs(confDiff) > 0.05) return confDiff;
      // Fall back to recency
      const aTime = new Date(a.last_confirmed || a.updated_at || 0).getTime();
      const bTime = new Date(b.last_confirmed || b.updated_at || 0).getTime();
      return bTime - aTime;
    });
  }, [patterns]);

  // Loading state
  if (loading && patterns.length === 0) {
    return (
      <div className="memory-subview" id="panel-patterns" role="tabpanel">
        <div className="patterns-status">
          <Loader size={24} className="patterns-status__spinner" aria-hidden="true" />
          <span className="patterns-status__text">Loading insights…</span>
        </div>
      </div>
    );
  }

  // Error state
  if (error && patterns.length === 0) {
    return (
      <div className="memory-subview" id="panel-patterns" role="tabpanel">
        <div className="patterns-status patterns-status--error">
          <AlertCircle size={24} aria-hidden="true" />
          <span className="patterns-status__text">Failed to load patterns</span>
          <span className="patterns-status__detail">{error}</span>
        </div>
      </div>
    );
  }

  // Empty state
  if (patterns.length === 0 && !loading) {
    return (
      <div className="memory-subview" id="panel-patterns" role="tabpanel">
        <div className="memory-subview__empty empty-state">
          <Sparkles size={40} className="empty-state__icon" aria-hidden="true" />
          <h3 className="empty-state__title">No Patterns Yet</h3>
          <p className="empty-state__description">
            Keep capturing blurts — patterns will emerge as the system learns
            your rhythms, preferences, and habits.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-subview" id="panel-patterns" role="tabpanel">
      {/* Summary header */}
      <div className="patterns-header">
        <div className="patterns-header__title-row">
          <Sparkles size={18} className="patterns-header__icon" aria-hidden="true" />
          <h2 className="patterns-header__title">
            {sortedPatterns.length} Insight{sortedPatterns.length !== 1 ? 's' : ''}
          </h2>
          {loading && (
            <Loader size={14} className="patterns-header__loading" aria-label="Refreshing" />
          )}
        </div>
        <p className="patterns-header__subtitle">
          Confirm patterns that resonate, dismiss ones that don't.
        </p>
      </div>

      {/* Type filter chips */}
      <div className="patterns-filters" role="group" aria-label="Filter by pattern type">
        <Filter size={14} className="patterns-filters__icon" aria-hidden="true" />
        <div className="patterns-filters__chips">
          {PATTERN_TYPES.map((pt) => (
            <button
              key={pt.id}
              className={`patterns-filter-chip ${typeFilter === pt.id ? 'patterns-filter-chip--active' : ''}`}
              onClick={() => handleTypeFilter(pt.id)}
              aria-pressed={typeFilter === pt.id}
            >
              {pt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Insight cards list */}
      <div className="patterns-list">
        {sortedPatterns.map((pattern) => (
          <InsightCard
            key={pattern.id}
            pattern={pattern}
            onConfirm={reinforcePattern}
            onDismiss={dismissPattern}
          />
        ))}
      </div>
    </div>
  );
}
