/**
 * JsonViewer — Syntax-highlighted, collapsible JSON response viewer.
 *
 * Features:
 *  - Recursive tree rendering with collapsible objects/arrays
 *  - Syntax highlighting by value type (string, number, boolean, null)
 *  - Copy-to-clipboard with visual feedback
 *  - Expand all / collapse all controls
 *  - Loading spinner and error states with retry
 *  - Graceful handling of non-JSON (renders as raw string)
 *
 * Props:
 *  - data: any          — parsed JSON data to render
 *  - title: string      — optional label in the toolbar
 *  - loading: boolean   — show loading spinner
 *  - error: string|null — show error state with message
 *  - onRetry: function  — callback for retry button in error state
 *  - defaultExpanded: boolean — start fully expanded (default: true)
 *  - maxDepth: number   — auto-collapse nodes deeper than this (default: 4)
 */
import { useState, useCallback, useMemo } from 'react';
import {
  ChevronDown,
  Copy,
  Check,
  AlertCircle,
  ChevronsDownUp,
  ChevronsUpDown,
  RefreshCw,
} from 'lucide-react';
import './JsonViewer.css';

/* ── helpers ───────────────────────────────────────────────────────── */

function typeOf(val) {
  if (val === null) return 'null';
  if (Array.isArray(val)) return 'array';
  return typeof val;
}

function itemCount(val) {
  if (Array.isArray(val)) return val.length;
  if (val && typeof val === 'object') return Object.keys(val).length;
  return 0;
}

function isExpandable(val) {
  return (
    val !== null &&
    typeof val === 'object' &&
    (Array.isArray(val) ? val.length > 0 : Object.keys(val).length > 0)
  );
}

/* ── JsonNode — recursive tree node ────────────────────────────────── */

function JsonNode({ keyName, value, depth, isLast, expandedMap, onToggle, path }) {
  const type = typeOf(value);
  const expandable = isExpandable(value);
  const nodePath = path;
  const expanded = expandedMap[nodePath] !== false; // default expanded

  const handleToggle = useCallback(() => {
    onToggle(nodePath);
  }, [nodePath, onToggle]);

  // Indent guides
  const indents = [];
  for (let i = 0; i < depth; i++) {
    indents.push(<span key={i} className="json-viewer__indent" aria-hidden="true" />);
  }

  const comma = isLast ? null : <span className="json-viewer__comma">,</span>;

  /* ── Leaf values ────────────────────────────────────────────────── */
  if (!expandable) {
    let display;
    switch (type) {
      case 'string':
        display = (
          <span className="json-viewer__value--string">
            &quot;{value}&quot;
          </span>
        );
        break;
      case 'number':
        display = <span className="json-viewer__value--number">{String(value)}</span>;
        break;
      case 'boolean':
        display = (
          <span className="json-viewer__value--boolean">{String(value)}</span>
        );
        break;
      case 'null':
        display = <span className="json-viewer__value--null">null</span>;
        break;
      default:
        display = <span className="json-viewer__value--string">{String(value)}</span>;
    }

    return (
      <div className="json-viewer__row">
        {indents}
        <span className="json-viewer__toggle-spacer" aria-hidden="true" />
        {keyName !== undefined && (
          <>
            <span className="json-viewer__key">&quot;{keyName}&quot;</span>
            <span className="json-viewer__colon">:&nbsp;</span>
          </>
        )}
        {display}
        {comma}
      </div>
    );
  }

  /* ── Expandable object / array ──────────────────────────────────── */
  const isArray = Array.isArray(value);
  const openBracket = isArray ? '[' : '{';
  const closeBracket = isArray ? ']' : '}';
  const entries = isArray ? value.map((v, i) => [i, v]) : Object.entries(value);
  const count = entries.length;

  // Empty containers
  if (count === 0) {
    return (
      <div className="json-viewer__row">
        {indents}
        <span className="json-viewer__toggle-spacer" aria-hidden="true" />
        {keyName !== undefined && (
          <>
            <span className="json-viewer__key">&quot;{keyName}&quot;</span>
            <span className="json-viewer__colon">:&nbsp;</span>
          </>
        )}
        <span className="json-viewer__bracket">{openBracket}{closeBracket}</span>
        {comma}
      </div>
    );
  }

  if (!expanded) {
    return (
      <div className="json-viewer__row">
        {indents}
        <button
          className="json-viewer__toggle json-viewer__toggle--collapsed"
          onClick={handleToggle}
          aria-label="Expand"
          aria-expanded="false"
        >
          <ChevronDown size={14} />
        </button>
        {keyName !== undefined && (
          <>
            <span className="json-viewer__key">&quot;{keyName}&quot;</span>
            <span className="json-viewer__colon">:&nbsp;</span>
          </>
        )}
        <span
          className="json-viewer__bracket json-viewer__bracket--collapsed"
          onClick={handleToggle}
        >
          {openBracket}
        </span>
        <span className="json-viewer__ellipsis">…</span>
        <span className="json-viewer__bracket">{closeBracket}</span>
        <span className="json-viewer__item-count">{count} {count === 1 ? 'item' : 'items'}</span>
        {comma}
      </div>
    );
  }

  return (
    <>
      {/* Opening line */}
      <div className="json-viewer__row">
        {indents}
        <button
          className="json-viewer__toggle"
          onClick={handleToggle}
          aria-label="Collapse"
          aria-expanded="true"
        >
          <ChevronDown size={14} />
        </button>
        {keyName !== undefined && (
          <>
            <span className="json-viewer__key">&quot;{keyName}&quot;</span>
            <span className="json-viewer__colon">:&nbsp;</span>
          </>
        )}
        <span className="json-viewer__bracket">{openBracket}</span>
      </div>

      {/* Children */}
      {entries.map(([k, v], idx) => (
        <JsonNode
          key={isArray ? idx : k}
          keyName={isArray ? undefined : k}
          value={v}
          depth={depth + 1}
          isLast={idx === count - 1}
          expandedMap={expandedMap}
          onToggle={onToggle}
          path={`${nodePath}.${k}`}
        />
      ))}

      {/* Closing line */}
      <div className="json-viewer__row">
        {indents}
        <span className="json-viewer__toggle-spacer" aria-hidden="true" />
        <span className="json-viewer__bracket">{closeBracket}</span>
        {comma}
      </div>
    </>
  );
}

/* ── Main component ────────────────────────────────────────────────── */

export default function JsonViewer({
  data,
  title,
  loading = false,
  error = null,
  onRetry,
  defaultExpanded = true,
  maxDepth = 4,
}) {
  // Track collapsed paths — by default everything is expanded
  // When defaultExpanded=false or depth > maxDepth, nodes start collapsed
  const [expandedMap, setExpandedMap] = useState(() => {
    if (defaultExpanded) return {};
    // Auto-collapse nodes beyond maxDepth
    return buildCollapsedMap(data, maxDepth);
  });

  const [copied, setCopied] = useState(false);

  const handleToggle = useCallback((path) => {
    setExpandedMap((prev) => ({
      ...prev,
      [path]: prev[path] === false ? true : false,
    }));
  }, []);

  const handleCopy = useCallback(async () => {
    try {
      const text = JSON.stringify(data, null, 2);
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = JSON.stringify(data, null, 2);
      textarea.style.position = 'fixed';
      textarea.style.opacity = '0';
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [data]);

  const handleExpandAll = useCallback(() => {
    setExpandedMap({});
  }, []);

  const handleCollapseAll = useCallback(() => {
    if (data != null) {
      setExpandedMap(buildCollapsedMap(data, 0));
    }
  }, [data]);

  // Memoize tree for perf
  const tree = useMemo(() => {
    if (data === undefined || data === null) return null;
    return (
      <JsonNode
        value={data}
        depth={0}
        isLast
        expandedMap={expandedMap}
        onToggle={handleToggle}
        path="$"
      />
    );
  }, [data, expandedMap, handleToggle]);

  /* ── Render ──────────────────────────────────────────────────────── */
  return (
    <div className="json-viewer" role="region" aria-label={title || 'JSON response'}>
      {/* Toolbar */}
      <div className="json-viewer__toolbar">
        {title && <span className="json-viewer__title">{title}</span>}
        {!title && <span />}
        <div className="json-viewer__actions">
          {data != null && !loading && !error && (
            <>
              <button
                className="json-viewer__action-btn"
                onClick={handleCollapseAll}
                title="Collapse all"
                aria-label="Collapse all"
              >
                <ChevronsDownUp size={16} />
              </button>
              <button
                className="json-viewer__action-btn"
                onClick={handleExpandAll}
                title="Expand all"
                aria-label="Expand all"
              >
                <ChevronsUpDown size={16} />
              </button>
              <button
                className={`json-viewer__action-btn ${copied ? 'json-viewer__action-btn--copied' : ''}`}
                onClick={handleCopy}
                title={copied ? 'Copied!' : 'Copy JSON'}
                aria-label={copied ? 'Copied to clipboard' : 'Copy JSON to clipboard'}
              >
                {copied ? <Check size={16} /> : <Copy size={16} />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="json-viewer__loading" role="status" aria-label="Loading">
          <div className="json-viewer__spinner" />
          <span className="json-viewer__loading-text">Fetching response…</span>
        </div>
      )}

      {/* Error state */}
      {!loading && error && (
        <div className="json-viewer__error" role="alert">
          <AlertCircle size={28} className="json-viewer__error-icon" />
          <p className="json-viewer__error-title">Request failed</p>
          <p className="json-viewer__error-message">{error}</p>
          {onRetry && (
            <button className="json-viewer__retry-btn" onClick={onRetry}>
              <RefreshCw size={14} style={{ marginRight: 6 }} />
              Retry
            </button>
          )}
        </div>
      )}

      {/* Empty state */}
      {!loading && !error && data == null && (
        <div className="json-viewer__empty">
          No data to display. Select an endpoint to fetch.
        </div>
      )}

      {/* JSON tree */}
      {!loading && !error && data != null && (
        <div className="json-viewer__content">
          <div className="json-viewer__node" role="tree">
            {tree}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Build a map of paths → false for auto-collapsing ──────────────── */
function buildCollapsedMap(data, maxDepth, path = '$', depth = 0) {
  let map = {};
  if (data === null || typeof data !== 'object') return map;

  if (depth >= maxDepth && isExpandable(data)) {
    map[path] = false;
  }

  const entries = Array.isArray(data)
    ? data.map((v, i) => [i, v])
    : Object.entries(data);

  for (const [k, v] of entries) {
    if (v !== null && typeof v === 'object') {
      Object.assign(map, buildCollapsedMap(v, maxDepth, `${path}.${k}`, depth + 1));
    }
  }

  return map;
}
