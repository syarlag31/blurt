/**
 * IntentBadge — visual label/tag showing the classified intent of a message.
 *
 * Displays a compact colored pill with an SVG icon and label text.
 * Uses INTENT_CONFIG from constants for color mapping.
 * Replaces emoji icons with Lucide SVG icons for premium feel.
 */
import {
  CheckSquare,
  Calendar,
  Bell,
  Lightbulb,
  BookOpen,
  RefreshCw,
  HelpCircle,
} from 'lucide-react';
import { INTENT_CONFIG } from '../utils/constants';

/** Map intent keys to Lucide icon components */
const INTENT_ICONS = {
  TASK: CheckSquare,
  EVENT: Calendar,
  REMINDER: Bell,
  IDEA: Lightbulb,
  JOURNAL: BookOpen,
  UPDATE: RefreshCw,
  QUESTION: HelpCircle,
};

/**
 * @param {object} props
 * @param {string} props.intent - intent string (e.g. "task", "IDEA", "reminder")
 */
export function IntentBadge({ intent }) {
  if (!intent) return null;

  const key = intent.toUpperCase();
  const config = INTENT_CONFIG[key];

  if (!config) return null;

  const IconComponent = INTENT_ICONS[key] || CheckSquare;

  return (
    <span
      className="intent-badge"
      style={{
        '--intent-color': config.color,
        '--intent-bg': `${config.color}1a`, /* ~10% opacity */
      }}
      aria-label={`Intent: ${config.label}`}
    >
      <IconComponent size={11} strokeWidth={2.5} className="intent-badge__icon" />
      <span className="intent-badge__label">{config.label}</span>
    </span>
  );
}
