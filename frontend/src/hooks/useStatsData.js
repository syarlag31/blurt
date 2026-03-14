/**
 * useStatsData — Fetches and manages real-time stats from all backend
 * health/stats endpoints. Supports:
 *   - Initial fetch on mount
 *   - Periodic polling (30s default)
 *   - External trigger re-fetch (e.g. from WebSocket blurt.created)
 *   - Per-category loading and error states
 *
 * Categories:
 *   1. health     — GET /health
 *   2. classify   — GET /api/v1/classify/health + /api/v1/classify/stats
 *   3. capture    — GET /api/v1/blurt/stats
 *   4. recall     — GET /api/v1/recall/stats
 *   5. calendar   — GET /auth/google/status
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE } from '../utils/constants';

const POLL_INTERVAL_MS = 30_000;

/** Initial state for a single stat category. */
function initCategory() {
  return { data: null, loading: true, error: null, lastFetched: null };
}

/** Safe JSON fetch with timeout. Returns { data, error }. */
async function safeFetch(url, timeoutMs = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) {
      return { data: null, error: `HTTP ${res.status}` };
    }
    const json = await res.json();
    return { data: json, error: null };
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      return { data: null, error: 'Timeout' };
    }
    return { data: null, error: err.message || 'Network error' };
  }
}

/**
 * @param {number} [refreshKey=0] — increment to force re-fetch (e.g. on blurt.created)
 * @param {boolean} [enabled=true] — set to false to pause polling
 */
export function useStatsData(refreshKey = 0, enabled = true) {
  const [stats, setStats] = useState({
    health: initCategory(),
    classify: initCategory(),
    classifyStats: initCategory(),
    capture: initCategory(),
    recall: initCategory(),
    calendar: initCategory(),
  });

  const mountedRef = useRef(true);
  const pollTimerRef = useRef(null);

  /** Fetch a single category and update state. */
  const fetchCategory = useCallback(async (key, url) => {
    // Set loading for this category (keep existing data visible)
    setStats((prev) => ({
      ...prev,
      [key]: { ...prev[key], loading: true, error: null },
    }));

    const { data, error } = await safeFetch(url);

    if (!mountedRef.current) return;

    setStats((prev) => ({
      ...prev,
      [key]: {
        data: data ?? prev[key].data, // keep stale data on error
        loading: false,
        error,
        lastFetched: error ? prev[key].lastFetched : Date.now(),
      },
    }));
  }, []);

  /** Fetch all categories in parallel. */
  const fetchAll = useCallback(() => {
    fetchCategory('health', '/health');
    fetchCategory('classify', `${API_BASE}/classify/health`);
    fetchCategory('classifyStats', `${API_BASE}/classify/stats`);
    fetchCategory('capture', `${API_BASE}/blurt/stats`);
    fetchCategory('recall', `${API_BASE}/recall/stats`);
    fetchCategory('calendar', '/auth/google/status');
  }, [fetchCategory]);

  // Initial fetch + re-fetch on refreshKey change
  useEffect(() => {
    if (!enabled) return;
    fetchAll();
  }, [fetchAll, refreshKey, enabled]);

  // Periodic polling
  useEffect(() => {
    if (!enabled) return;
    pollTimerRef.current = setInterval(fetchAll, POLL_INTERVAL_MS);
    return () => clearInterval(pollTimerRef.current);
  }, [fetchAll, enabled]);

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  return { stats, refetch: fetchAll };
}
