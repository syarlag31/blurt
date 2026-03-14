/**
 * useQuestionAnswer — hook to fetch question answers from the backend.
 *
 * When a message is classified as QUESTION intent, this hook calls
 * POST /api/v1/question to get a synthesized answer with source references.
 *
 * Returns:
 * - answer: the QuestionAPIResponse object (or null)
 * - loading: whether the fetch is in progress
 * - error: error message if fetch failed
 */
import { useState, useEffect, useRef } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';

/**
 * @param {string|null} query - The question text to send to the backend
 * @param {boolean} enabled - Whether to actually fetch (prevents firing for non-QUESTION intents)
 * @returns {{ answer: object|null, loading: boolean, error: string|null }}
 */
export function useQuestionAnswer(query, enabled = false) {
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fetchedRef = useRef(null);

  useEffect(() => {
    if (!enabled || !query || fetchedRef.current === query) return;
    fetchedRef.current = query;

    let cancelled = false;
    const fetchAnswer = async () => {
      setLoading(true);
      setError(null);

      try {
        const res = await fetch(`${API_BASE}/question`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: USER_ID,
            query,
            tier: 'premium', // dogfooding UI gets full access
          }),
        });

        if (!res.ok) {
          throw new Error(`Question API returned ${res.status}`);
        }

        const data = await res.json();
        if (!cancelled) {
          setAnswer(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message || 'Failed to fetch answer');
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchAnswer();

    return () => {
      cancelled = true;
    };
  }, [query, enabled]);

  return { answer, loading, error };
}
