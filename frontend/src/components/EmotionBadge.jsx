/**
 * EmotionBadge — visual tag showing the detected emotion of an episode.
 *
 * Displays a compact colored pill with a dot indicator and label.
 * Uses EMOTION_COLORS from constants for color mapping.
 */
import { EMOTION_COLORS } from '../utils/constants';

/**
 * @param {object} props
 * @param {string} props.emotion - primary emotion (e.g. "joy", "trust")
 * @param {number} [props.intensity] - 0-3 intensity scale
 */
export function EmotionBadge({ emotion, intensity }) {
  if (!emotion) return null;

  const key = emotion.toLowerCase();
  const color = EMOTION_COLORS[key] || '#8494a7';

  // Capitalize first letter
  const label = key.charAt(0).toUpperCase() + key.slice(1);

  return (
    <span
      className="emotion-badge"
      style={{
        '--emotion-color': color,
        '--emotion-bg': `${color}1a`, /* ~10% opacity */
      }}
      aria-label={`Emotion: ${label}${intensity != null ? ` (${intensity.toFixed(1)})` : ''}`}
    >
      <span className="emotion-badge__dot" aria-hidden="true" />
      <span className="emotion-badge__label">{label}</span>
    </span>
  );
}
