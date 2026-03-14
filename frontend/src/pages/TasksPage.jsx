/**
 * TasksPage — Task surfacing and management tab.
 *
 * Fetches ranked tasks from the Thompson Sampling surfacing API,
 * renders them in a scrollable list of SwipeableTaskCard components, and
 * handles feedback actions (complete, defer, drop) via buttons or swipe gestures.
 *
 * Swipe right → complete | Swipe left → defer | Buttons for all 3 actions.
 * Shows UndoToast after swipe actions with 5-second undo window.
 *
 * States: loading → empty | error | task-list
 * Re-fetches when navigated to or when blurt.created arrives.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { CheckCircle, Loader2, AlertCircle, Inbox } from 'lucide-react';
import { SwipeableTaskCard } from '../components/SwipeableTaskCard';
import { UndoToast } from '../components/UndoToast';
import { useBroadcastRefresh } from '../hooks/useBroadcastRefresh';
import { API_BASE, USER_ID } from '../utils/constants';
import './TasksPage.css';

/** Map API response task → shape expected by TaskCard */
function mapTask(t) {
  return {
    id: t.task_id,
    content: t.content,
    intent: t.intent,
    score: t.composite_score,
    reason: t.surfacing_reason,
    status: t.status,
    entity_names: t.entity_names,
    project: t.project,
    due_at: t.due_at,
    estimated_energy: t.estimated_energy,
    times_surfaced: t.times_surfaced,
    created_at: t.created_at,
  };
}

export default function TasksPage() {
  const [tasks, setTasks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null); // response metadata
  const [actioningId, setActioningId] = useState(null); // task currently being actioned
  const [undoState, setUndoState] = useState(null); // { taskId, action, task, message }
  const abortRef = useRef(null);
  const pendingActionRef = useRef(null); // holds the pending API call timer for undo
  const refreshKey = useBroadcastRefresh();

  const fetchTasks = useCallback(async () => {
    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      const url = `${API_BASE}/tasks/surface?user_id=${encodeURIComponent(USER_ID)}&max_results=10`;
      const res = await fetch(url, { signal: controller.signal });

      if (!res.ok) {
        throw new Error(`Server error (${res.status})`);
      }

      const data = await res.json();
      const mapped = (data.tasks || []).map(mapTask);
      setTasks(mapped);
      setMeta({
        total_in_store: data.total_in_store,
        total_eligible: data.total_eligible,
        message: data.message,
      });
    } catch (err) {
      if (err.name === 'AbortError') return; // cancelled — ignore
      setError(err.message || 'Failed to load tasks');
    } finally {
      if (!controller.signal.aborted) {
        setLoading(false);
      }
    }
  }, []);

  // Fetch on mount + cleanup
  useEffect(() => {
    fetchTasks();
    return () => {
      if (abortRef.current) abortRef.current.abort();
      if (pendingActionRef.current) clearTimeout(pendingActionRef.current);
    };
  }, [fetchTasks]);

  // Re-fetch when blurt.created arrives via BroadcastChannel
  const refreshKeyRef = useRef(refreshKey);
  useEffect(() => {
    // Skip the initial mount (already handled above)
    if (refreshKeyRef.current === refreshKey) return;
    refreshKeyRef.current = refreshKey;
    fetchTasks();
  }, [refreshKey, fetchTasks]);

  /** Fire the feedback API call (best-effort) and re-fetch. */
  const sendFeedback = useCallback(
    async (action, taskId) => {
      try {
        await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: USER_ID, action }),
        });
      } catch {
        // Best effort — still remove card for responsive feel
      }
      // Re-fetch after a brief delay for scoring to settle
      setTimeout(() => fetchTasks(), 400);
    },
    [fetchTasks],
  );

  /** Handle button-tap feedback (immediate, no undo). */
  const handleAction = useCallback(
    async (action, taskId) => {
      setActioningId(taskId);

      // Optimistically remove the task from the list
      setTasks((prev) => prev.filter((t) => t.id !== taskId));
      setActioningId(null);

      await sendFeedback(action, taskId);
    },
    [sendFeedback],
  );

  /** Handle swipe-to-dismiss (optimistic removal + undo toast).
   *  Note: SwipeableTaskCard already submits feedback to the API,
   *  so we only handle UI removal, undo restore, and re-fetch here. */
  const handleSwipeAction = useCallback(
    (action, taskId) => {
      // Find the task being dismissed (for undo restore)
      const dismissedTask = tasks.find((t) => t.id === taskId);
      const actionLabel = action === 'complete' ? 'completed' : action === 'engage' ? 'engaged' : 'deferred';

      // Optimistically remove the task
      setTasks((prev) => prev.filter((t) => t.id !== taskId));

      // Clear any existing undo / pending re-fetch
      if (pendingActionRef.current) {
        clearTimeout(pendingActionRef.current);
        pendingActionRef.current = null;
      }

      // Schedule a re-fetch after the undo window closes
      pendingActionRef.current = setTimeout(() => {
        fetchTasks();
        pendingActionRef.current = null;
      }, 5200);

      // Show undo toast
      setUndoState({
        taskId,
        action,
        task: dismissedTask,
        message: `Task ${actionLabel}`,
      });
    },
    [tasks, fetchTasks],
  );

  /** Restore task on undo. */
  const handleUndo = useCallback(() => {
    if (!undoState?.task) return;

    // Cancel the pending API call
    if (pendingActionRef.current) {
      clearTimeout(pendingActionRef.current);
      pendingActionRef.current = null;
    }

    // Restore the task to the list (in its original position by score)
    setTasks((prev) => {
      const restored = [...prev, undoState.task];
      restored.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
      return restored;
    });

    setUndoState(null);
  }, [undoState]);

  /** Dismiss undo toast (action proceeds). */
  const handleUndoDismiss = useCallback(() => {
    setUndoState(null);
  }, []);

  // Pull-to-refresh via manual button
  const handleRefresh = useCallback(() => {
    fetchTasks();
  }, [fetchTasks]);

  // ── Render states ─────────────────────────────────────────────

  // Loading state with skeleton cards
  if (loading && tasks.length === 0) {
    return (
      <div className="page page--tasks">
        <header className="tasks-header">
          <h2 className="tasks-header__title">Tasks</h2>
        </header>
        <div className="tasks-loading" role="status" aria-label="Loading tasks">
          <div className="tasks-skeleton">
            <div className="tasks-skeleton__card skeleton" />
            <div className="tasks-skeleton__card skeleton" />
            <div className="tasks-skeleton__card skeleton" />
          </div>
          <p className="tasks-loading__text">
            <Loader2 size={16} className="tasks-loading__spinner" />
            Surfacing tasks…
          </p>
        </div>
      </div>
    );
  }

  // Error state
  if (error && tasks.length === 0) {
    return (
      <div className="page page--tasks">
        <header className="tasks-header">
          <h2 className="tasks-header__title">Tasks</h2>
        </header>
        <div className="tasks-error" role="alert">
          <AlertCircle size={40} className="tasks-error__icon" />
          <p className="tasks-error__title">Couldn't load tasks</p>
          <p className="tasks-error__detail">{error}</p>
          <button className="tasks-error__retry" onClick={handleRefresh}>
            Try again
          </button>
        </div>
      </div>
    );
  }

  // Empty state — shame-free messaging
  if (!loading && tasks.length === 0) {
    return (
      <div className="page page--tasks">
        <header className="tasks-header">
          <h2 className="tasks-header__title">Tasks</h2>
        </header>
        <div className="tasks-empty">
          <div className="tasks-empty__icon-wrap">
            <CheckCircle size={48} className="tasks-empty__icon" />
          </div>
          <p className="tasks-empty__title">
            {meta?.message || 'All clear'}
          </p>
          <p className="tasks-empty__desc">
            No tasks need your attention right now. Enjoy the moment.
          </p>
          <button className="tasks-empty__refresh" onClick={handleRefresh}>
            Refresh
          </button>
        </div>
      </div>
    );
  }

  // Task list
  return (
    <div className="page page--tasks">
      <header className="tasks-header">
        <h2 className="tasks-header__title">Tasks</h2>
        <div className="tasks-header__meta">
          {meta && meta.total_in_store != null && (
            <span className="tasks-header__count">
              {tasks.length} of {meta.total_in_store}
            </span>
          )}
          <button
            className="tasks-header__refresh"
            onClick={handleRefresh}
            disabled={loading}
            aria-label="Refresh tasks"
            title="Refresh"
          >
            <Loader2
              size={18}
              className={`tasks-header__refresh-icon${loading ? ' tasks-header__refresh-icon--spinning' : ''}`}
            />
          </button>
        </div>
      </header>

      {error && (
        <div className="tasks-inline-error" role="alert">
          <AlertCircle size={14} />
          <span>{error}</span>
        </div>
      )}

      <div className="tasks-list" role="list">
        {tasks.map((task) => (
          <div
            className={`tasks-list__item${actioningId === task.id ? ' tasks-list__item--exiting' : ''}`}
            role="listitem"
            key={task.id}
          >
            <SwipeableTaskCard
              task={task}
              onAction={handleAction}
              onSwipeAction={handleSwipeAction}
            />
          </div>
        ))}
      </div>

      {/* Undo toast for swipe actions */}
      {undoState && (
        <UndoToast
          message={undoState.message}
          onUndo={handleUndo}
          onDismiss={handleUndoDismiss}
        />
      )}

      {!loading && tasks.length > 0 && (
        <>
          <p className="tasks-swipe-hint">
            Swipe right to complete · Swipe left to defer
          </p>
          <p className="tasks-footer">
            <Inbox size={14} />
            {meta?.total_eligible != null
              ? `${meta.total_eligible} eligible task${meta.total_eligible !== 1 ? 's' : ''} in store`
              : 'Ranked by Thompson Sampling'}
          </p>
        </>
      )}
    </div>
  );
}
