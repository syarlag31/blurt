/**
 * Persistent top card showing the single best-ranked task via Thompson Sampling.
 *
 * Always visible at the top of the screen (sticky). Displays:
 * - Intent badge with color coding
 * - Composite score from Thompson Sampling surfacing engine
 * - Task content
 * - Surfacing reason (why this task was ranked #1)
 * - Project / entity context (if available)
 * - Shame-free action buttons: Done, Later, Let go
 *
 * When isNudge is true, applies a CSS animation for visual highlight.
 * Parent controls animation lifecycle via key prop for re-triggering.
 */
import { INTENT_CONFIG } from '../utils/constants';

export function TaskCard({ task, onAction, isNudge }) {
  if (!task) {
    return (
      <div className="task-card task-card--empty">
        <p className="task-card__empty-text">No tasks right now. Enjoy the moment.</p>
      </div>
    );
  }

  const intent = INTENT_CONFIG[task.intent?.toUpperCase()] || INTENT_CONFIG.TASK;
  const hasContext = task.project || (task.entity_names && task.entity_names.length > 0);

  return (
    <div
      className={`task-card${isNudge ? ' task-card--nudge' : ''}`}
      role="region"
      aria-label="Top task"
    >
      <div className="task-card__header">
        <span className="task-card__intent" style={{ backgroundColor: intent.color }}>
          {intent.icon} {intent.label}
        </span>
        {task.score != null && (
          <span
            className="task-card__score"
            title="Thompson Sampling composite score"
          >
            {Math.round(task.score * 100)}%
          </span>
        )}
      </div>

      <p className="task-card__content">{task.content}</p>

      {task.reason && <p className="task-card__reason">{task.reason}</p>}

      {hasContext && (
        <div className="task-card__context">
          {task.project && (
            <span className="task-card__tag task-card__tag--project">{task.project}</span>
          )}
          {task.entity_names &&
            task.entity_names.map((name) => (
              <span className="task-card__tag" key={name}>
                {name}
              </span>
            ))}
        </div>
      )}

      <div className="task-card__actions">
        <button
          className="task-card__btn task-card__btn--complete"
          onClick={() => onAction?.('complete', task.id)}
          aria-label="Complete task"
        >
          Done
        </button>
        <button
          className="task-card__btn task-card__btn--defer"
          onClick={() => onAction?.('defer', task.id)}
          aria-label="Defer task"
        >
          Later
        </button>
        <button
          className="task-card__btn task-card__btn--drop"
          onClick={() => onAction?.('drop', task.id)}
          aria-label="Drop task"
        >
          Let go
        </button>
      </div>
    </div>
  );
}
