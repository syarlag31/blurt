/**
 * EpisodeCard — renders a single episode from episodic memory.
 *
 * Displays:
 *   - Intent badge (task, event, reminder, etc.)
 *   - Emotion badge (joy, trust, fear, etc.)
 *   - Raw text (clamped to 3 lines)
 *   - Entity chips
 *   - Timestamp and modality indicator
 *
 * Supports both regular episodes and compressed summary cards.
 */
import { Mic, Keyboard, Layers } from 'lucide-react';
import { IntentBadge } from '../IntentBadge';
import { EmotionBadge } from '../EmotionBadge';
import { EntityChip } from '../EntityChip';

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

const MODALITY_ICONS = {
  voice: Mic,
  text: Keyboard,
};

/**
 * @param {object} props
 * @param {object} props.episode - episode data from the API
 */
export function EpisodeCard({ episode }) {
  if (!episode) return null;

  const ModIcon = MODALITY_ICONS[episode.modality] || Keyboard;

  return (
    <article className="episode-card" aria-label="Episode">
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

/**
 * SummaryCard — renders a compressed episode summary.
 */
export function SummaryCard({ summary }) {
  if (!summary) return null;

  const topIntents = Object.entries(summary.intent_distribution || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const topEntities = Object.entries(summary.entity_mentions || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4);

  return (
    <article className="episode-card episode-card--summary" aria-label="Episode summary">
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

      <div className="episode-card__summary-meta">
        {topIntents.map(([intent, count]) => (
          <span key={intent} className="episode-card__summary-stat">
            <IntentBadge intent={intent} />
            <span>{count}</span>
          </span>
        ))}
      </div>

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
