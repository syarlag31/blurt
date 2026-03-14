/**
 * MetadataCard — Expanded detail panel beneath chat messages.
 *
 * Displays classification result with premium visual treatment:
 * - Intent badge with icon
 * - Confidence score as visual progress bar
 * - Extracted entities as styled chips/tags
 * - Detected emotion with color-coded icon
 * - Full classification scores breakdown (expanded view)
 * - Transcript and latency details
 *
 * Tap to expand/collapse with smooth animation.
 */
import { useState } from 'react';
import {
  ChevronDown,
  Heart,
  Shield,
  AlertTriangle,
  Zap,
  Cloud,
  Frown,
  Flame,
  Sparkles,
  Tag,
  Brain,
  Clock,
  BarChart3,
  FileText,
} from 'lucide-react';
import { INTENT_CONFIG, EMOTION_COLORS } from '../utils/constants';
import './MetadataCard.css';

/** Emotion icon mapping — Lucide SVG icons, no emojis */
const EMOTION_ICONS = {
  joy: Sparkles,
  trust: Shield,
  fear: AlertTriangle,
  surprise: Zap,
  sadness: Cloud,
  disgust: Frown,
  anger: Flame,
  anticipation: Heart,
};

/**
 * Normalize emotion value — may be a string or object with {primary, intensity, valence, arousal}.
 * @param {*} emotion
 * @returns {{ label: string, intensity?: number, valence?: number, arousal?: number } | null}
 */
function normalizeEmotion(emotion) {
  if (!emotion) return null;
  if (typeof emotion === 'string') return { label: emotion };
  if (emotion.primary) {
    return {
      label: emotion.primary,
      intensity: emotion.intensity,
      valence: emotion.valence,
      arousal: emotion.arousal,
    };
  }
  return null;
}

/**
 * Confidence bar — visual indicator with color gradient.
 */
function ConfidenceBar({ value }) {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  // Color based on confidence level
  const color =
    pct >= 80
      ? 'var(--success, #10b981)'
      : pct >= 50
        ? 'var(--warning, #f59e0b)'
        : 'var(--danger, #ef4444)';

  return (
    <div className="mc-confidence" aria-label={`Confidence: ${pct}%`}>
      <div className="mc-confidence__track">
        <div
          className="mc-confidence__fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="mc-confidence__label">{pct}%</span>
    </div>
  );
}

/**
 * Entity chip — styled tag with icon.
 */
function EntityChip({ name }) {
  return (
    <span className="mc-entity-chip">
      <Tag size={10} strokeWidth={2.5} className="mc-entity-chip__icon" />
      <span className="mc-entity-chip__text">{name}</span>
    </span>
  );
}

/**
 * Emotion indicator — icon + label with semantic color.
 */
function EmotionIndicator({ emotion }) {
  if (!emotion) return null;
  const color = EMOTION_COLORS[emotion.label] || 'var(--text-secondary)';
  const IconComp = EMOTION_ICONS[emotion.label] || Heart;

  return (
    <span className="mc-emotion" style={{ '--emotion-color': color }}>
      <span className="mc-emotion__icon-wrap">
        <IconComp size={12} strokeWidth={2.5} />
      </span>
      <span className="mc-emotion__label">{emotion.label}</span>
      {emotion.intensity != null && (
        <span className="mc-emotion__intensity">
          {Math.round(emotion.intensity * 100)}%
        </span>
      )}
    </span>
  );
}

/**
 * Score bar — horizontal bar for each classification score.
 */
function ScoreBar({ label, value, isTop }) {
  const pct = Math.round(value * 100);
  return (
    <div className={`mc-score-bar ${isTop ? 'mc-score-bar--top' : ''}`}>
      <span className="mc-score-bar__label">{label}</span>
      <div className="mc-score-bar__track">
        <div
          className="mc-score-bar__fill"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="mc-score-bar__value">{pct}%</span>
    </div>
  );
}

export function MetadataCard({ metadata }) {
  const [expanded, setExpanded] = useState(false);

  if (!metadata) return null;

  const intent =
    INTENT_CONFIG[metadata.intent?.toUpperCase()] || INTENT_CONFIG.TASK;
  const confidence = metadata.confidence;
  const emotion = normalizeEmotion(metadata.emotion);

  // Normalize entities
  const entityNames = (metadata.entities || []).map((e) =>
    typeof e === 'string' ? e : e?.name || String(e),
  );

  // Sort scores descending for the expanded classification breakdown
  const sortedScores = metadata.all_scores
    ? Object.entries(metadata.all_scores).sort(
        ([, a], [, b]) => Number(b) - Number(a),
      )
    : [];

  const topScoreKey = sortedScores.length > 0 ? sortedScores[0][0] : null;

  return (
    <div
      className={`mc ${expanded ? 'mc--expanded' : ''}`}
      onClick={() => setExpanded((e) => !e)}
      role="button"
      tabIndex={0}
      aria-expanded={expanded}
      onKeyDown={(e) => e.key === 'Enter' && setExpanded((v) => !v)}
    >
      {/* ── Summary row ─────────────────────────────────────────── */}
      <div className="mc__summary">
        <span className="mc__intent" style={{ '--intent-color': intent.color }}>
          {intent.label}
        </span>

        <ConfidenceBar value={confidence} />

        {emotion && <EmotionIndicator emotion={emotion} />}

        <span
          className={`mc__chevron expandable-chevron${expanded ? ' expandable-chevron--open' : ''}`}
          aria-hidden="true"
        >
          <ChevronDown size={14} strokeWidth={2.5} />
        </span>
      </div>

      {/* ── Entity chips (always visible if present) ────────────── */}
      {entityNames.length > 0 && (
        <div className="mc__entities">
          {entityNames.map((name, i) => (
            <EntityChip key={i} name={name} />
          ))}
        </div>
      )}

      {/* ── Expanded details panel ──────────────────────────────── */}
      <div className={`expandable-section${expanded ? ' expandable-section--open' : ''}`}>
        <div className="expandable-section__inner">
          <div className="expandable-section__content">
        <div className="mc__details">
          {/* Emotion detail with valence/arousal */}
          {emotion && (emotion.valence != null || emotion.arousal != null) && (
            <div className="mc__section">
              <div className="mc__section-header">
                <Heart size={12} strokeWidth={2.5} />
                <span>Emotion Detail</span>
              </div>
              <div className="mc__emotion-detail">
                {emotion.valence != null && (
                  <div className="mc__emotion-metric">
                    <span className="mc__emotion-metric-label">Valence</span>
                    <div className="mc__emotion-metric-bar">
                      <div
                        className="mc__emotion-metric-fill mc__emotion-metric-fill--valence"
                        style={{
                          width: `${Math.abs(emotion.valence) * 50}%`,
                          marginLeft:
                            emotion.valence >= 0 ? '50%' : `${50 - Math.abs(emotion.valence) * 50}%`,
                        }}
                      />
                      <div className="mc__emotion-metric-center" />
                    </div>
                    <span className="mc__emotion-metric-value">
                      {emotion.valence > 0 ? '+' : ''}
                      {emotion.valence.toFixed(2)}
                    </span>
                  </div>
                )}
                {emotion.arousal != null && (
                  <div className="mc__emotion-metric">
                    <span className="mc__emotion-metric-label">Arousal</span>
                    <div className="mc__emotion-metric-bar">
                      <div
                        className="mc__emotion-metric-fill mc__emotion-metric-fill--arousal"
                        style={{ width: `${emotion.arousal * 100}%` }}
                      />
                    </div>
                    <span className="mc__emotion-metric-value">
                      {emotion.arousal.toFixed(2)}
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Classification scores breakdown */}
          {sortedScores.length > 0 && (
            <div className="mc__section">
              <div className="mc__section-header">
                <BarChart3 size={12} strokeWidth={2.5} />
                <span>Classification Scores</span>
              </div>
              <div className="mc__scores">
                {sortedScores.map(([key, val]) => (
                  <ScoreBar
                    key={key}
                    label={key}
                    value={Number(val)}
                    isTop={key === topScoreKey}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Transcript */}
          {metadata.transcript && (
            <div className="mc__section">
              <div className="mc__section-header">
                <FileText size={12} strokeWidth={2.5} />
                <span>Transcript</span>
              </div>
              <p className="mc__transcript">{metadata.transcript}</p>
            </div>
          )}

          {/* Latency */}
          {metadata.latency_ms != null && (
            <div className="mc__section mc__section--inline">
              <Clock size={11} strokeWidth={2.5} />
              <span className="mc__latency-label">Latency</span>
              <span className="mc__latency-value">
                {Math.round(metadata.latency_ms)}ms
              </span>
            </div>
          )}

          {/* Raw data hint */}
          {metadata.intent && (
            <div className="mc__section mc__section--inline">
              <Brain size={11} strokeWidth={2.5} />
              <span className="mc__raw-label">Intent</span>
              <code className="mc__raw-value">{metadata.intent}</code>
            </div>
          )}
        </div>
          </div>
        </div>
      </div>
    </div>
  );
}
