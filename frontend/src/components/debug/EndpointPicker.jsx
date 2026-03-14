/**
 * EndpointPicker — Collapsible accordion of all backend API endpoints.
 *
 * Groups endpoints by router category. Each section expands/collapses
 * to reveal individual endpoints with method badge, path, and description.
 * Tapping an endpoint fires the onSelect callback.
 */

import { useState, useCallback } from 'react';
import { ChevronDown, Server, Zap } from 'lucide-react';
import { ENDPOINT_GROUPS, TOTAL_ENDPOINTS, METHOD_COLORS } from '../../utils/endpointRegistry.js';
import './EndpointPicker.css';

/** Color for HTTP method badges */
function methodColor(method) {
  return METHOD_COLORS[method] || 'var(--text-secondary)';
}

/** Single endpoint row */
function EndpointRow({ endpoint, onSelect }) {
  const handleClick = useCallback(() => {
    onSelect?.(endpoint);
  }, [endpoint, onSelect]);

  return (
    <button
      className="ep-row"
      onClick={handleClick}
      type="button"
      aria-label={`${endpoint.method} ${endpoint.path}`}
    >
      <span
        className="ep-row__method"
        style={{ '--method-color': methodColor(endpoint.method) }}
      >
        {endpoint.method}
      </span>
      <div className="ep-row__info">
        <span className="ep-row__label">{endpoint.label}</span>
        <span className="ep-row__path">{endpoint.path}</span>
      </div>
    </button>
  );
}

/** Accordion section for a single router group */
function EndpointSection({ group, isOpen, onToggle, onSelect }) {
  const handleToggle = useCallback(() => {
    onToggle(group.key);
  }, [group.key, onToggle]);

  return (
    <div className={`ep-section ${isOpen ? 'ep-section--open' : ''}`}>
      <button
        className="ep-section__header"
        onClick={handleToggle}
        type="button"
        aria-expanded={isOpen}
        aria-controls={`ep-section-${group.key}`}
      >
        <div className="ep-section__header-left">
          <span className="ep-section__title">{group.label}</span>
          <span className="ep-section__count">{group.endpoints.length}</span>
        </div>
        <ChevronDown
          size={18}
          className={`ep-section__chevron ${isOpen ? 'ep-section__chevron--open' : ''}`}
          aria-hidden="true"
        />
      </button>

      <div
        id={`ep-section-${group.key}`}
        className="ep-section__body"
        role="region"
        aria-label={`${group.label} endpoints`}
      >
        <div className="ep-section__list">
          {group.endpoints.map((ep) => (
            <EndpointRow
              key={`${ep.method}-${ep.path}`}
              endpoint={ep}
              onSelect={onSelect}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

/** Main endpoint picker with global expand/collapse */
export default function EndpointPicker({ onSelect, selectedEndpoint }) {
  const [openSections, setOpenSections] = useState(new Set());

  const toggleSection = useCallback((key) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setOpenSections(new Set(ENDPOINT_GROUPS.map((g) => g.key)));
  }, []);

  const collapseAll = useCallback(() => {
    setOpenSections(new Set());
  }, []);

  const allOpen = openSections.size === ENDPOINT_GROUPS.length;

  return (
    <div className="ep-picker">
      {/* Header */}
      <div className="ep-picker__header">
        <div className="ep-picker__header-left">
          <Server size={16} className="ep-picker__icon" aria-hidden="true" />
          <span className="ep-picker__title">API Endpoints</span>
          <span className="ep-picker__badge">{TOTAL_ENDPOINTS}</span>
        </div>
        <button
          className="ep-picker__toggle-all"
          onClick={allOpen ? collapseAll : expandAll}
          type="button"
        >
          {allOpen ? 'Collapse all' : 'Expand all'}
        </button>
      </div>

      {/* Selected indicator */}
      {selectedEndpoint && (
        <div className="ep-picker__selected">
          <Zap size={14} aria-hidden="true" />
          <span
            className="ep-picker__selected-method"
            style={{ '--method-color': methodColor(selectedEndpoint.method) }}
          >
            {selectedEndpoint.method}
          </span>
          <span className="ep-picker__selected-label">{selectedEndpoint.label}</span>
        </div>
      )}

      {/* Accordion sections */}
      <div className="ep-picker__sections">
        {ENDPOINT_GROUPS.map((group) => (
          <EndpointSection
            key={group.key}
            group={group}
            isOpen={openSections.has(group.key)}
            onToggle={toggleSection}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}
