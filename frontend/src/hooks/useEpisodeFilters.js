/**
 * useEpisodeFilters — Filter state management and filtered episode fetching.
 *
 * Manages intent, emotion, entity, and date range filters with debounced
 * API queries to GET /api/v1/episodes/user/{user_id}.
 *
 * Returns filter state, setters, computed query params, fetched episodes,
 * and loading/error state.
 */
import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';

const DEFAULT_LIMIT = 50;

/** Build query string from active filters. */
function buildQueryParams(filters) {
  const params = new URLSearchParams();

  if (filters.intent) params.set('intent', filters.intent);
  if (filters.emotion) params.set('emotion', filters.emotion);
  if (filters.entity) params.set('entity', filters.entity);
  if (filters.dateStart) params.set('start', filters.dateStart);
  if (filters.dateEnd) params.set('end', filters.dateEnd);
  if (filters.minIntensity > 0) params.set('min_intensity', String(filters.minIntensity));

  params.set('limit', String(filters.limit || DEFAULT_LIMIT));
  params.set('offset', String(filters.offset || 0));

  return params.toString();
}

/** Initial empty filter state. */
const INITIAL_FILTERS = {
  intent: '',
  emotion: '',
  entity: '',
  dateStart: '',
  dateEnd: '',
  minIntensity: 0,
  limit: DEFAULT_LIMIT,
  offset: 0,
};

export function useEpisodeFilters(refreshKey = 0) {
  const [filters, setFilters] = useState(INITIAL_FILTERS);
  const [episodes, setEpisodes] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [totalHint, setTotalHint] = useState(0);
  const abortRef = useRef(null);

  /** Count of active filters (for badge display). */
  const activeFilterCount = useMemo(() => {
    let count = 0;
    if (filters.intent) count++;
    if (filters.emotion) count++;
    if (filters.entity) count++;
    if (filters.dateStart || filters.dateEnd) count++;
    return count;
  }, [filters.intent, filters.emotion, filters.entity, filters.dateStart, filters.dateEnd]);

  /** Set a single filter field and reset offset to 0. */
  const setFilter = useCallback((field, value) => {
    setFilters((prev) => ({
      ...prev,
      [field]: value,
      offset: field === 'offset' ? value : 0,
    }));
  }, []);

  /** Clear all filters back to initial state. */
  const clearFilters = useCallback(() => {
    setFilters(INITIAL_FILTERS);
  }, []);

  /** The query string for the current filter state. */
  const queryString = useMemo(() => buildQueryParams(filters), [filters]);

  /** Fetch episodes whenever filters or refreshKey change. */
  useEffect(() => {
    // Abort any pending request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const url = `${API_BASE}/episodes/user/${USER_ID}?${queryString}`;

    setLoading(true);
    setError(null);

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        // Backend may return { episodes: [...] } or a plain array
        const list = Array.isArray(data) ? data : (data.episodes ?? data.items ?? []);
        setEpisodes(list);
        setTotalHint(data.total ?? list.length);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        setError(err.message);
        setLoading(false);
      });

    return () => controller.abort();
  }, [queryString, refreshKey]);

  return {
    filters,
    setFilter,
    clearFilters,
    activeFilterCount,
    episodes,
    loading,
    error,
    totalHint,
  };
}
