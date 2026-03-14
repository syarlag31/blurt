/**
 * QuestionAnswerCard — Displays synthesized answer for QUESTION intent messages.
 *
 * Shows:
 * - The answer summary text (from backend or WebSocket response.text)
 * - Tappable source reference chips that expand to show episode content
 * - Result count and query type badge
 * - Loading shimmer while fetching from the question API
 * - Confidence scores for source results (when available)
 *
 * Design: Premium card with subtle border, smooth expand transitions,
 * and touch-friendly source chips (min 44px tap targets).
 */
import { useState, useCallback } from 'react';
import {
  HelpCircle,
  BookOpen,
  ChevronRight,
  ExternalLink,
  Loader2,
  AlertCircle,
  Hash,
  Sparkles,
} from 'lucide-react';
import { useQuestionAnswer } from '../hooks/useQuestionAnswer';
import './QuestionAnswerCard.css';

/**
 * Format a query type string for display.
 * @param {string} queryType - e.g. "entity_lookup", "semantic_recall"
 * @returns {string} e.g. "Entity Lookup", "Semantic Recall"
 */
function formatQueryType(queryType) {
  if (!queryType) return 'Question';
  return queryType
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Truncate text to a maximum length with ellipsis.
 */
function truncate(text, maxLen = 120) {
  if (!text || text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + '…';
}

/**
 * SourceChip — tappable reference to a source episode/result.
 * Expands on tap to show full content.
 */
function SourceChip({ result, index, expanded, onTap }) {
  const content = result.content || result.name || 'Source';
  const type = result.type || result.fact_type || '';
  const similarity = result.similarity ?? result.relevance_score;
  const hasDetails = result.metadata || result.source_context || similarity != null;

  return (
    <button
      className={`qa-source ${expanded ? 'qa-source--expanded' : ''}`}
      onClick={(e) => {
        e.stopPropagation();
        onTap(index);
      }}
      aria-expanded={expanded}
      aria-label={`Source ${index + 1}: ${truncate(content, 40)}`}
    >
      <div className="qa-source__header">
        <span className="qa-source__index">
          <Hash size={10} strokeWidth={2.5} />
          {index + 1}
        </span>
        <span className="qa-source__preview">
          {expanded ? content : truncate(content, 60)}
        </span>
        <ChevronRight
          size={14}
          strokeWidth={2}
          className={`qa-source__chevron ${expanded ? 'qa-source__chevron--open' : ''}`}
        />
      </div>

      {expanded && hasDetails && (
        <div className="qa-source__details">
          {type && (
            <span className="qa-source__type-badge">{type}</span>
          )}
          {similarity != null && (
            <span className="qa-source__score">
              <Sparkles size={10} strokeWidth={2.5} />
              {Math.round(similarity * 100)}% match
            </span>
          )}
          {result.timestamp && (
            <span className="qa-source__time">
              {new Date(result.timestamp).toLocaleDateString([], {
                month: 'short',
                day: 'numeric',
              })}
            </span>
          )}
          {result.source_context && (
            <p className="qa-source__context">
              {truncate(
                typeof result.source_context === 'string'
                  ? result.source_context
                  : JSON.stringify(result.source_context),
                200,
              )}
            </p>
          )}
        </div>
      )}
    </button>
  );
}

/**
 * Loading shimmer placeholder while fetching answer.
 */
function AnswerSkeleton() {
  return (
    <div className="qa-card qa-card--loading" aria-busy="true">
      <div className="qa-card__header">
        <Loader2 size={16} strokeWidth={2} className="qa-card__spinner" />
        <span className="qa-card__loading-text">Finding answer…</span>
      </div>
      <div className="qa-card__shimmer">
        <div className="qa-shimmer-line qa-shimmer-line--long" />
        <div className="qa-shimmer-line qa-shimmer-line--medium" />
        <div className="qa-shimmer-line qa-shimmer-line--short" />
      </div>
    </div>
  );
}

/**
 * QuestionAnswerCard component.
 *
 * @param {object} props
 * @param {string} props.questionText - The original question text
 * @param {string} [props.wsAnswerText] - Answer text received via WebSocket (immediate)
 * @param {boolean} [props.autoFetch=true] - Whether to auto-fetch from /api/v1/question
 */
export function QuestionAnswerCard({ questionText, wsAnswerText, autoFetch = true }) {
  const [expandedSource, setExpandedSource] = useState(null);

  // Fetch detailed answer from the question API
  const { answer: apiAnswer, loading, error } = useQuestionAnswer(
    questionText,
    autoFetch,
  );

  const toggleSource = useCallback((index) => {
    setExpandedSource((prev) => (prev === index ? null : index));
  }, []);

  // Use API answer summary if available, else fall back to WebSocket answer
  const answerText = apiAnswer?.answer_summary || wsAnswerText || null;
  const results = apiAnswer?.results || [];
  const queryType = apiAnswer?.query_type;
  const resultCount = apiAnswer?.result_count ?? results.length;
  const totalAvailable = apiAnswer?.total_available ?? 0;
  const sourceEpisodes = apiAnswer?.source_episodes || [];
  const upgradeHint = apiAnswer?.upgrade_hint;

  // Show loading state while fetching
  if (loading && !wsAnswerText) {
    return <AnswerSkeleton />;
  }

  // If we have neither API answer nor WS answer, show nothing
  if (!answerText && results.length === 0 && !error) {
    // But if we have a WS answer text, show it minimally
    if (wsAnswerText) {
      return (
        <div className="qa-card">
          <div className="qa-card__header">
            <HelpCircle size={16} strokeWidth={2} className="qa-card__icon" />
            <span className="qa-card__title">Answer</span>
          </div>
          <p className="qa-card__answer">{wsAnswerText}</p>
        </div>
      );
    }
    return null;
  }

  // Error state — still show WS answer if available
  if (error && !answerText && results.length === 0) {
    return (
      <div className="qa-card qa-card--error">
        <div className="qa-card__header">
          <AlertCircle size={16} strokeWidth={2} className="qa-card__icon qa-card__icon--error" />
          <span className="qa-card__title">Answer</span>
        </div>
        {wsAnswerText && <p className="qa-card__answer">{wsAnswerText}</p>}
        <p className="qa-card__error-text">{error}</p>
      </div>
    );
  }

  return (
    <div className="qa-card">
      {/* Header with query type badge */}
      <div className="qa-card__header">
        <HelpCircle size={16} strokeWidth={2} className="qa-card__icon" />
        <span className="qa-card__title">Answer</span>
        {queryType && (
          <span className="qa-card__query-type">{formatQueryType(queryType)}</span>
        )}
        {resultCount > 0 && (
          <span className="qa-card__count">
            {resultCount}{totalAvailable > resultCount ? ` of ${totalAvailable}` : ''} source{resultCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Synthesized answer text */}
      {answerText && (
        <p className="qa-card__answer">{answerText}</p>
      )}

      {/* Loading indicator while API is fetching but we have WS answer */}
      {loading && wsAnswerText && (
        <div className="qa-card__fetching">
          <Loader2 size={12} strokeWidth={2} className="qa-card__spinner qa-card__spinner--small" />
          <span>Loading sources…</span>
        </div>
      )}

      {/* Source references — tappable chips */}
      {results.length > 0 && (
        <div className="qa-card__sources">
          <div className="qa-card__sources-header">
            <BookOpen size={12} strokeWidth={2.5} />
            <span>Sources</span>
          </div>
          <div className="qa-card__source-list" role="list">
            {results.map((result, i) => (
              <SourceChip
                key={result.id || `src-${i}`}
                result={result}
                index={i}
                expanded={expandedSource === i}
                onTap={toggleSource}
              />
            ))}
          </div>
        </div>
      )}

      {/* Source episode IDs — compact reference list */}
      {sourceEpisodes.length > 0 && (
        <div className="qa-card__episode-refs">
          <ExternalLink size={10} strokeWidth={2.5} />
          <span>{sourceEpisodes.length} episode{sourceEpisodes.length !== 1 ? 's' : ''} referenced</span>
        </div>
      )}

      {/* Upgrade hint — anti-shame messaging */}
      {upgradeHint && (
        <p className="qa-card__upgrade-hint">{upgradeHint}</p>
      )}
    </div>
  );
}
