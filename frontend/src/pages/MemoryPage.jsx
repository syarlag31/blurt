/**
 * MemoryPage — Knowledge graph and entity exploration.
 *
 * Container page with segmented pill control for 4 sub-views:
 *   Timeline | Patterns | Rhythms | Graph
 *
 * Uses useState for active sub-view management with conditional
 * rendering of the appropriate sub-view component.
 *
 * The FilterBar + useEpisodeFilters hook provide intent, emotion,
 * entity, and date range filtering with live-query to the episodes API.
 */
import { useState, useCallback } from 'react';
import { Clock, Sparkles, Activity, Share2 } from 'lucide-react';
import { SegmentedPills } from '../components/SegmentedPills';
import FilterBar from '../components/memory/FilterBar';
import { useEpisodeFilters } from '../hooks/useEpisodeFilters';
import { useBroadcastRefresh } from '../hooks/useBroadcastRefresh';
import TimelineView from '../components/memory/TimelineView';
import PatternsView from '../components/memory/PatternsView';
import RhythmsView from '../components/memory/RhythmsView';
import GraphView from '../components/memory/GraphView';
import './MemoryPage.css';

/** Segment definitions for the pill control */
const MEMORY_SEGMENTS = [
  { id: 'timeline', label: 'Timeline', icon: Clock },
  { id: 'patterns', label: 'Patterns', icon: Sparkles },
  { id: 'rhythms',  label: 'Rhythms',  icon: Activity },
  { id: 'graph',    label: 'Graph',    icon: Share2 },
];

/** Map segment IDs to their view components */
const VIEW_MAP = {
  timeline: TimelineView,
  patterns: PatternsView,
  rhythms:  RhythmsView,
  graph:    GraphView,
};

export default function MemoryPage() {
  const [activeView, setActiveView] = useState('timeline');
  const refreshKey = useBroadcastRefresh();

  const {
    filters,
    setFilter,
    clearFilters,
    activeFilterCount,
    episodes,
    loading,
    error,
    totalHint,
  } = useEpisodeFilters(refreshKey);

  const handleViewChange = useCallback((id) => {
    setActiveView(id);
  }, []);

  const ActiveComponent = VIEW_MAP[activeView];

  /** Props forwarded to sub-views for filtered data access. */
  const dataProps = { episodes, loading, error, totalHint, filters, refreshKey };

  return (
    <div className="page page--memory">
      <header className="memory-header">
        <h1 className="memory-header__title">Memory</h1>
      </header>

      <div className="memory-pills-wrapper">
        <SegmentedPills
          segments={MEMORY_SEGMENTS}
          activeId={activeView}
          onChange={handleViewChange}
        />
      </div>

      {/* Filter bar — visible on timeline/patterns views */}
      {(activeView === 'timeline' || activeView === 'patterns') && (
        <FilterBar
          filters={filters}
          setFilter={setFilter}
          clearFilters={clearFilters}
          activeFilterCount={activeFilterCount}
        />
      )}

      <div className="memory-content">
        <ActiveComponent {...dataProps} />
      </div>
    </div>
  );
}
