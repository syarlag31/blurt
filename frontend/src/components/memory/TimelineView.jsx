/**
 * TimelineView — Reverse-chronological episode list with semantic search.
 *
 * Default view: fetches recent episodes from GET /api/v1/episodes/user/{user_id}
 * Search mode: uses POST /api/v1/recall for natural-language semantic search
 *
 * Episodes are grouped by day with intent/emotion/entity badges.
 * Supports both raw episodes and compressed summaries.
 * Automatically refreshes when blurt.created arrives via BroadcastChannel.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  Clock, Search, X, Loader, AlertCircle, Zap,
  FileText, User, Lightbulb, Hash, Mic, Keyboard, Layers,
} from 'lucide-react';
import { USER_ID, API_BASE, INTENT_CONFIG, EMOTION_COLORS } from '../../utils/constants';
import { useBroadcastRefresh } from '../../hooks/useBroadcastRefresh';
import { IntentBadge } from '../IntentBadge';
import { EmotionBadge } from '../EmotionBadge';
import { EntityChip } from '../EntityChip';
import './TimelineView.css';

const PAGE_SIZE = 50;

/* ── Source type icon mapping (for search results) ──────────── */
const SOURCE_ICONS = {
  episode: FileText,
  entity: User,
  fact: Lightbulb,
  pattern: Hash,
  summary: FileText,
};

/* ── Helpers ──────────────────────────────────────────────────── */

/** Safe JSON fetch with timeout + abort support. */
async function safeFetch(url, options = {}, timeoutMs = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return { data: null, error: `HTTP ${res.status}`, status: res.status };
    const json = await res.json();
    return { data: json, error: null, status: res.status };
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') return { data: null, error: 'Request timed out', status: 0 };
    return { data: null, error: err.message || 'Network error', status: 0 };
  }
}

/** Format ISO timestamp to human-readable relative time. */
function formatRelativeTime(isoString) {
  if (!isoString) return '';
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now - date;
  const diffMin = Math.floor(diffMs / 60000);
  const diffHr = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHr < 24) return `${diffHr}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

/** Format ISO timestamp to display time (e.g. "2:35 PM") */
function formatTime(isoString) {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  } catch {
    return '';
  }
}

/** Group episodes by day label (Today, Yesterday, Mar 10, etc.) */
function groupByDay(episodes) {
  const groups = [];
  let currentLabel = null;
  let currentItems = [];

  const now = new Date();
  const todayStr = now.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toDateString();

  for (const ep of episodes) {
    const ts = ep.timestamp;
    if (!ts) continue;
    const d = new Date(ts);
    const dayStr = d.toDateString();

    let label;
    if (dayStr === todayStr) {
      label = 'Today';
    } else if (dayStr === yesterdayStr) {
      label = 'Yesterday';
    } else {
      label = d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: d.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
      });
    }

    if (label !== currentLabel) {
      if (currentItems.length > 0) {
        groups.push({ label: currentLabel, items: currentItems });
      }
      currentLabel = label;
      currentItems = [ep];
    } else {
      currentItems.push(ep);
    }
  }

  if (currentItems.length > 0) {
    groups.push({ label: currentLabel, items: currentItems });
  }

  return groups;
}

/* ── Modality icons ───────────────────────────────────────────── */
const MODALITY_ICONS = {
  voice: Mic,
  text: Keyboard,
};

/* ── Episode card with intent/emotion/entity badges ──────────── */
function EpisodeCard({ episode }) {
  const ModIcon = MODALITY_ICONS[episode.modality] || Keyboard;

  return (
    <article className="episode-card" role="article">
      {/* Header: intent + emotion badges, time */}
      <div className="episode-card__header">
        <div className="episode-card__badges">
          <IntentBadge intent={episode.intent} />
          {episode.emotion?.primary && (
            <EmotionBadge
              emotion={episode.emotion.primary}
              intensity={episode.emotion.intensity}
            />
          )}
        </div>
        <time className="episode-card__time" dateTime={episode.timestamp}>
          {formatTime(episode.timestamp)}
        </time>
      </div>

      {/* Body text */}
      <p className="episode-card__text">{episode.raw_text}</p>

      {/* Footer: entities + modality */}
      {(episode.entities?.length > 0 || episode.modality) && (
        <div className="episode-card__footer">
          <div className="episode-card__entities">
            {(episode.entities || []).slice(0, 4).map((ent, i) => (
              <EntityChip
                key={ent.entity_id || `${ent.name}-${i}`}
                name={ent.name}
                entityType={ent.entity_type}
              />
            ))}
          </div>
          <span className="episode-card__modality">
            <ModIcon size={12} strokeWidth={2} aria-hidden="true" />
            {episode.modality}
          </span>
        </div>
      )}
    </article>
  );
}

/* ── Compressed summary card ─────────────────────────────────── */
function SummaryCard({ summary }) {
  if (!summary) return null;

  const topIntents = Object.entries(summary.intent_distribution || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const topEntities = Object.entries(summary.entity_mentions || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return (
    <article className="episode-card episode-card--summary" role="article">
      <div className="episode-card__header">
        <span className="episode-card__summary-label">
          <Layers size={12} strokeWidth={2.5} aria-hidden="true" />
          Summary
        </span>
        <span className="episode-card__time">
          {summary.episode_count} episode{summary.episode_count !== 1 ? 's' : ''}
        </span>
      </div>

      <p className="episode-card__text">{summary.summary_text}</p>

      {topIntents.length > 0 && (
        <div className="episode-card__summary-meta">
          {topIntents.map(([intent, count]) => (
            <span key={intent} className="episode-card__summary-stat">
              <IntentBadge intent={intent} />
              <span>{count}</span>
            </span>
          ))}
        </div>
      )}

      {topEntities.length > 0 && (
        <div className="episode-card__entities">
          {topEntities.map(([name, count]) => (
            <EntityChip key={name} name={`${name} (${count})`} />
          ))}
        </div>
      )}
    </article>
  );
}

/* ── Search result card (recall mode) ────────────────────────── */
function SearchResultCard({ result }) {
  const IconComponent = SOURCE_ICONS[result.source_type] || FileText;
  return (
    <article className="episode-card episode-card--search" role="article">
      <div className="episode-card__header">
        <span className="episode-card__source-type">
          <IconComponent size={14} aria-hidden="true" />
          {result.source_type}
        </span>
        <div className="episode-card__header-right">
          <RelevanceBadge score={result.relevance_score} />
          {result.timestamp && (
            <time className="episode-card__time" dateTime={result.timestamp}>
              {formatRelativeTime(result.timestamp)}
            </time>
          )}
        </div>
      </div>
      <p className="episode-card__text">{result.content}</p>
      {result.source_context?.surrounding_entities?.length > 0 && (
        <div className="episode-card__entities">
          {result.source_context.surrounding_entities.map((name, i) => (
            <EntityChip key={`${name}-${i}`} name={name} />
          ))}
        </div>
      )}
    </article>
  );
}

/** Format a relevance score as a percentage badge. */
function RelevanceBadge({ score }) {
  if (score == null) return null;
  const pct = Math.round(score * 100);
  const cls =
    pct >= 80 ? 'timeline-relevance--high' :
    pct >= 50 ? 'timeline-relevance--mid' :
    'timeline-relevance--low';
  return (
    <span className={`timeline-relevance ${cls}`} aria-label={`${pct}% relevant`}>
      {pct}%
    </span>
  );
}

/* ── Main component ──────────────────────────────────────────── */
export default function TimelineView() {
  const refreshKey = useBroadcastRefresh();

  // Search state
  const [query, setQuery] = useState('');
  const [searchActive, setSearchActive] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);
  const [searchMeta, setSearchMeta] = useState(null);

  // Episode list state (default view)
  const [episodes, setEpisodes] = useState([]);
  const [episodesLoading, setEpisodesLoading] = useState(true);
  const [episodesError, setEpisodesError] = useState(null);
  const [totalCount, setTotalCount] = useState(0);

  const searchInputRef = useRef(null);
  const debounceRef = useRef(null);

  /* ── Fetch episodes (default timeline) ───────────────────── */
  const fetchEpisodes = useCallback(async () => {
    setEpisodesLoading(true);

    // Try recall endpoint first (returns episodes + summaries)
    const recallResult = await safeFetch(
      `${API_BASE}/episodes/recall/${encodeURIComponent(USER_ID)}?limit=${PAGE_SIZE}&include_compressed=true`
    );

    if (!recallResult.error && recallResult.data) {
      // Recall returns { entries: [{ entry_type, episode, summary, timestamp }], ... }
      const entries = recallResult.data.entries || [];
      // Flatten entries into a unified list
      setEpisodes(entries);
      setTotalCount(recallResult.data.total_count || entries.length);
      setEpisodesError(null);
      setEpisodesLoading(false);
      return;
    }

    // Fallback to plain episodes endpoint
    const { data, error } = await safeFetch(
      `${API_BASE}/episodes/user/${encodeURIComponent(USER_ID)}?limit=${PAGE_SIZE}`
    );
    if (error) {
      setEpisodesError(error);
    } else {
      // Wrap in recall-like format for consistent rendering
      const eps = (data.episodes || []).map(ep => ({
        entry_type: 'episode',
        episode: ep,
        timestamp: ep.timestamp,
      }));
      setEpisodes(eps);
      setTotalCount(data.total_count || 0);
      setEpisodesError(null);
    }
    setEpisodesLoading(false);
  }, []);

  // Initial load + refresh on blurt.created
  useEffect(() => {
    if (!searchActive) {
      fetchEpisodes();
    }
  }, [fetchEpisodes, refreshKey, searchActive]);

  /* ── Semantic search via recall endpoint ─────────────────── */
  const executeSearch = useCallback(async (q) => {
    if (!q.trim()) {
      setSearchActive(false);
      setSearchResults(null);
      setSearchMeta(null);
      return;
    }
    setSearchActive(true);
    setSearchLoading(true);
    setSearchError(null);

    const { data, error } = await safeFetch(`${API_BASE}/recall`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: USER_ID,
        query: q.trim(),
        max_results: 25,
      }),
    });

    if (error) {
      setSearchError(error);
      setSearchResults(null);
      setSearchMeta(null);
    } else {
      setSearchResults(data.results || []);
      setSearchMeta({
        totalResults: data.total_results,
        latencyMs: data.latency_ms,
        sourcesSearched: data.sources_searched,
        entityContext: data.entity_context_used,
        queryUnderstanding: data.query_understanding,
      });
      setSearchError(null);
    }
    setSearchLoading(false);
  }, []);

  /* ── Debounced search on input ───────────────────────────── */
  const handleInputChange = useCallback((e) => {
    const val = e.target.value;
    setQuery(val);
    clearTimeout(debounceRef.current);
    if (!val.trim()) {
      setSearchActive(false);
      setSearchResults(null);
      setSearchMeta(null);
      return;
    }
    debounceRef.current = setTimeout(() => {
      executeSearch(val);
    }, 400);
  }, [executeSearch]);

  /** Submit search on Enter. */
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      clearTimeout(debounceRef.current);
      executeSearch(query);
    }
  }, [executeSearch, query]);

  /** Clear search and return to default timeline. */
  const handleClear = useCallback(() => {
    setQuery('');
    setSearchActive(false);
    setSearchResults(null);
    setSearchError(null);
    setSearchMeta(null);
    searchInputRef.current?.focus();
  }, []);

  // Cleanup debounce on unmount
  useEffect(() => {
    return () => clearTimeout(debounceRef.current);
  }, []);

  /* ── Render helpers ─────────────────────────────────────── */
  const isLoading = searchActive ? searchLoading : episodesLoading;
  const currentError = searchActive ? searchError : episodesError;

  /** Render an entry from the recall response (episode or summary) */
  function renderEntry(entry, idx) {
    if (entry.entry_type === 'summary' && entry.summary) {
      return (
        <SummaryCard
          key={entry.summary.id || `summary-${idx}`}
          summary={entry.summary}
        />
      );
    }
    const ep = entry.episode || entry;
    return <EpisodeCard key={ep.id || `ep-${idx}`} episode={ep} />;
  }

  /** Build day-grouped timeline from entries */
  function renderTimeline() {
    // Entries might be recall-format (with entry_type) or plain episodes
    const items = episodes.map(e => ({
      ...e,
      _ts: e.timestamp || e.episode?.timestamp || e.summary?.created_at,
    }));

    const dayGroups = groupByDay(items.map(e => ({ ...e, timestamp: e._ts })));

    return dayGroups.map((group) => (
      <div className="timeline__day-group" key={group.label}>
        <div className="timeline__day-label">{group.label}</div>
        {group.items.map((item, idx) => renderEntry(item, idx))}
      </div>
    ));
  }

  /* ── Render ──────────────────────────────────────────────── */
  return (
    <div className="memory-subview" id="panel-timeline" role="tabpanel">
      {/* ── Search bar ──────────────────────────────────────── */}
      <div className="timeline-search">
        <div className="timeline-search__input-wrap">
          <Search
            size={18}
            className="timeline-search__icon"
            aria-hidden="true"
          />
          <input
            ref={searchInputRef}
            type="search"
            className="timeline-search__input"
            placeholder="Search your memory..."
            value={query}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            aria-label="Search memories"
            autoComplete="off"
            enterKeyHint="search"
          />
          {query && (
            <button
              className="timeline-search__clear"
              onClick={handleClear}
              aria-label="Clear search"
              type="button"
            >
              <X size={16} />
            </button>
          )}
        </div>
        {searchMeta && searchActive && !searchLoading && (
          <div className="timeline-search__meta" aria-live="polite">
            <span className="timeline-search__meta-count">
              {searchMeta.totalResults} result{searchMeta.totalResults !== 1 ? 's' : ''}
            </span>
            <span className="timeline-search__meta-latency">
              {Math.round(searchMeta.latencyMs)}ms
            </span>
            {searchMeta.entityContext?.length > 0 && (
              <span className="timeline-search__meta-entities">
                via {searchMeta.entityContext.join(', ')}
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Loading state ───────────────────────────────────── */}
      {isLoading && (
        <div className="timeline__loading" role="status" aria-label="Loading">
          <div className="timeline__spinner" aria-hidden="true" />
          <span className="timeline__loading-text">
            {searchActive ? 'Searching...' : 'Loading timeline...'}
          </span>
        </div>
      )}

      {/* ── Error state ─────────────────────────────────────── */}
      {currentError && !isLoading && (
        <div className="timeline__error" role="alert">
          <AlertCircle size={32} className="timeline__error-icon" aria-hidden="true" />
          <p className="timeline__error-text">{currentError}</p>
          <button
            className="timeline__retry-btn"
            onClick={searchActive ? () => executeSearch(query) : fetchEpisodes}
            type="button"
          >
            Retry
          </button>
        </div>
      )}

      {/* ── Search results ──────────────────────────────────── */}
      {searchActive && !searchLoading && !searchError && searchResults && (
        <div className="timeline" role="list">
          {searchResults.length === 0 ? (
            <div className="memory-subview__empty empty-state">
              <Search size={40} className="empty-state__icon" aria-hidden="true" />
              <h3 className="empty-state__title">No results</h3>
              <p className="empty-state__description">
                Try different keywords or a broader query.
              </p>
            </div>
          ) : (
            searchResults.map((result, i) => (
              <SearchResultCard key={`${result.source_id}-${i}`} result={result} />
            ))
          )}
        </div>
      )}

      {/* ── Default episode timeline (day-grouped) ──────────── */}
      {!searchActive && !episodesLoading && !episodesError && (
        <div className="timeline" role="list">
          {episodes.length === 0 ? (
            <div className="memory-subview__empty empty-state">
              <Clock size={40} className="empty-state__icon" aria-hidden="true" />
              <h3 className="empty-state__title">Timeline</h3>
              <p className="empty-state__description">
                Your memory timeline will appear here as you capture blurts.
              </p>
            </div>
          ) : (
            <>
              <div className="timeline__count">
                <Zap size={14} aria-hidden="true" />
                <span>{totalCount} episode{totalCount !== 1 ? 's' : ''}</span>
              </div>
              {renderTimeline()}
            </>
          )}
        </div>
      )}
    </div>
  );
}
