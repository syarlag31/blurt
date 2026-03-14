/**
 * usePatterns — Fetches learned patterns from the backend.
 *
 * Queries GET /api/v1/users/{user_id}/patterns and provides
 * reinforce/dismiss actions per pattern.
 *
 * Refreshes automatically when refreshKey changes (blurt.created).
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';

/**
 * @param {number} refreshKey — increment to trigger re-fetch
 * @param {object} [filterOpts] — optional { type, min_confidence }
 */
export function usePatterns(refreshKey = 0, filterOpts = {}) {
  const [patterns, setPatterns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const fetchPatterns = useCallback(() => {
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const params = new URLSearchParams();
    if (filterOpts.type) params.set('type', filterOpts.type);
    if (filterOpts.min_confidence) params.set('min_confidence', String(filterOpts.min_confidence));
    params.set('active', 'true');

    const qs = params.toString();
    const url = `${API_BASE}/users/${USER_ID}/patterns${qs ? `?${qs}` : ''}`;

    setLoading(true);
    setError(null);

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const list = Array.isArray(data) ? data : (data.patterns ?? data.items ?? []);
        setPatterns(list);
        setLoading(false);
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        setError(err.message);
        setLoading(false);
      });

    return () => controller.abort();
  }, [filterOpts.type, filterOpts.min_confidence]);

  useEffect(() => {
    const cleanup = fetchPatterns();
    return cleanup;
  }, [fetchPatterns, refreshKey]);

  /** Reinforce a pattern (confirm it). */
  const reinforcePattern = useCallback(async (patternId) => {
    try {
      const res = await fetch(
        `${API_BASE}/users/${USER_ID}/patterns/${patternId}/reinforce`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' } }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Optimistically update the local list
      setPatterns((prev) =>
        prev.map((p) =>
          p.id === patternId
            ? { ...p, confidence: Math.min(1, (p.confidence || 0.5) + 0.1), _confirmed: true }
            : p
        )
      );
      return true;
    } catch (err) {
      console.error('Failed to reinforce pattern:', err);
      return false;
    }
  }, []);

  /** Dismiss a pattern (deactivate). */
  const dismissPattern = useCallback(async (patternId) => {
    try {
      const res = await fetch(
        `${API_BASE}/users/${USER_ID}/patterns/${patternId}`,
        { method: 'DELETE' }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Optimistically remove from list
      setPatterns((prev) => prev.filter((p) => p.id !== patternId));
      return true;
    } catch (err) {
      console.error('Failed to dismiss pattern:', err);
      return false;
    }
  }, []);

  /** Weaken a pattern (soft disagree without full dismiss). */
  const weakenPattern = useCallback(async (patternId) => {
    try {
      const res = await fetch(
        `${API_BASE}/users/${USER_ID}/patterns/${patternId}/weaken`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' } }
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setPatterns((prev) =>
        prev.map((p) =>
          p.id === patternId
            ? { ...p, confidence: Math.max(0, (p.confidence || 0.5) - 0.1) }
            : p
        )
      );
      return true;
    } catch (err) {
      console.error('Failed to weaken pattern:', err);
      return false;
    }
  }, []);

  return {
    patterns,
    loading,
    error,
    reinforcePattern,
    dismissPattern,
    weakenPattern,
    refetch: fetchPatterns,
  };
}
