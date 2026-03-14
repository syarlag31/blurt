/**
 * Metadata card shown beneath user messages in the chat feed.
 *
 * Displays classification result: intent, entities, emotion, confidence.
 * Compact display that expands on tap for full details.
 */
import { useState } from 'react';
import { INTENT_CONFIG, EMOTION_COLORS } from '../utils/constants';

export function MetadataCard({ metadata }) {
  const [expanded, setExpanded] = useState(false);

  if (!metadata) return null;

  const intent = INTENT_CONFIG[metadata.intent?.toUpperCase()] || INTENT_CONFIG.TASK;
  const confidence = metadata.confidence != null ? Math.round(metadata.confidence * 100) : null;

  return (
    <div
      className="metadata-card"
      onClick={() => setExpanded((e) => !e)}
      role="button"
      tabIndex={0}
      aria-expanded={expanded}
      onKeyDown={(e) => e.key === 'Enter' && setExpanded((v) => !v)}
    >
      <div className="metadata-card__summary">
        <span className="metadata-card__intent" style={{ color: intent.color }}>
          {intent.icon} {intent.label}
        </span>

        {confidence != null && (
          <span className="metadata-card__confidence">{confidence}%</span>
        )}

        {metadata.entities?.length > 0 && (
          <span className="metadata-card__entities">
            {metadata.entities.slice(0, 3).join(', ')}
            {metadata.entities.length > 3 && ` +${metadata.entities.length - 3}`}
          </span>
        )}

        {metadata.emotion && (
          <span
            className="metadata-card__emotion"
            style={{ color: EMOTION_COLORS[metadata.emotion] || '#888' }}
          >
            {metadata.emotion}
          </span>
        )}
      </div>

      {expanded && (
        <div className="metadata-card__details">
          {metadata.transcript && (
            <div className="metadata-card__row">
              <span className="metadata-card__label">Transcript</span>
              <span className="metadata-card__value">{metadata.transcript}</span>
            </div>
          )}

          {metadata.entities?.length > 0 && (
            <div className="metadata-card__row">
              <span className="metadata-card__label">Entities</span>
              <span className="metadata-card__value">{metadata.entities.join(', ')}</span>
            </div>
          )}

          {metadata.all_scores && (
            <div className="metadata-card__row">
              <span className="metadata-card__label">Scores</span>
              <span className="metadata-card__value">
                {Object.entries(metadata.all_scores)
                  .map(([k, v]) => `${k}: ${Math.round(v * 100)}%`)
                  .join(', ')}
              </span>
            </div>
          )}

          {metadata.latency_ms != null && (
            <div className="metadata-card__row">
              <span className="metadata-card__label">Latency</span>
              <span className="metadata-card__value">{Math.round(metadata.latency_ms)}ms</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
