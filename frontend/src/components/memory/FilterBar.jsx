/**
 * FilterBar — Horizontally scrollable filter controls for episode queries.
 *
 * Provides chip-style filter toggles for intent, emotion, entity text,
 * and date range. Sits below the segmented pills in the Memory tab.
 *
 * Design: premium dark theme, 44px+ touch targets, Inter font, Lucide icons.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import {
  Filter,
  X,
  ChevronDown,
  Target,
  Heart,
  User,
  Calendar,
  Check,
} from 'lucide-react';
import { INTENT_CONFIG, EMOTION_COLORS } from '../../utils/constants';
import './FilterBar.css';

/** Intent options derived from constants. */
const INTENT_OPTIONS = Object.entries(INTENT_CONFIG).map(([key, cfg]) => ({
  value: key.toLowerCase(),
  label: cfg.label,
  color: cfg.color,
}));

/** Emotion options derived from constants. */
const EMOTION_OPTIONS = Object.entries(EMOTION_COLORS).map(([key, color]) => ({
  value: key,
  label: key.charAt(0).toUpperCase() + key.slice(1),
  color,
}));

/** Reusable dropdown for chip filters. */
function FilterDropdown({ options, value, onSelect, onClose, label }) {
  const ref = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (ref.current && !ref.current.contains(e.target)) {
        onClose();
      }
    }
    document.addEventListener('pointerdown', handleClickOutside);
    return () => document.removeEventListener('pointerdown', handleClickOutside);
  }, [onClose]);

  return (
    <div className="filter-dropdown" ref={ref} role="listbox" aria-label={label}>
      {options.map((opt) => (
        <button
          key={opt.value}
          className={`filter-dropdown__item ${value === opt.value ? 'filter-dropdown__item--active' : ''}`}
          role="option"
          aria-selected={value === opt.value}
          onClick={() => {
            onSelect(value === opt.value ? '' : opt.value);
            onClose();
          }}
        >
          <span
            className="filter-dropdown__dot"
            style={{ backgroundColor: opt.color }}
            aria-hidden="true"
          />
          <span className="filter-dropdown__label">{opt.label}</span>
          {value === opt.value && (
            <Check size={14} className="filter-dropdown__check" aria-hidden="true" />
          )}
        </button>
      ))}
    </div>
  );
}

/** Entity text input popover. */
function EntityInput({ value, onChange, onClose }) {
  const inputRef = useRef(null);
  const wrapperRef = useRef(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    function handleClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        onClose();
      }
    }
    document.addEventListener('pointerdown', handleClickOutside);
    return () => document.removeEventListener('pointerdown', handleClickOutside);
  }, [onClose]);

  return (
    <div className="filter-dropdown filter-dropdown--entity" ref={wrapperRef}>
      <div className="filter-entity-input-row">
        <input
          ref={inputRef}
          type="text"
          className="filter-entity-input"
          placeholder="Entity name…"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === 'Escape') onClose();
          }}
        />
        {value && (
          <button
            className="filter-entity-clear"
            onClick={() => { onChange(''); onClose(); }}
            aria-label="Clear entity filter"
          >
            <X size={14} />
          </button>
        )}
      </div>
    </div>
  );
}

/** Date range popover with start/end date inputs. */
function DateRangeInput({ dateStart, dateEnd, onChangeStart, onChangeEnd, onClose }) {
  const wrapperRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(e) {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        onClose();
      }
    }
    document.addEventListener('pointerdown', handleClickOutside);
    return () => document.removeEventListener('pointerdown', handleClickOutside);
  }, [onClose]);

  return (
    <div className="filter-dropdown filter-dropdown--dates" ref={wrapperRef}>
      <label className="filter-date-label">
        <span>From</span>
        <input
          type="date"
          className="filter-date-input"
          value={dateStart}
          onChange={(e) => onChangeStart(e.target.value ? new Date(e.target.value).toISOString() : '')}
        />
      </label>
      <label className="filter-date-label">
        <span>To</span>
        <input
          type="date"
          className="filter-date-input"
          value={dateEnd ? dateEnd.slice(0, 10) : ''}
          onChange={(e) => onChangeEnd(e.target.value ? new Date(e.target.value + 'T23:59:59').toISOString() : '')}
        />
      </label>
      <div className="filter-date-actions">
        <button
          className="filter-date-clear"
          onClick={() => { onChangeStart(''); onChangeEnd(''); onClose(); }}
        >
          Clear
        </button>
        <button className="filter-date-apply" onClick={onClose}>
          Apply
        </button>
      </div>
    </div>
  );
}

export default function FilterBar({ filters, setFilter, clearFilters, activeFilterCount }) {
  const [openDropdown, setOpenDropdown] = useState(null);

  const toggle = useCallback((name) => {
    setOpenDropdown((prev) => (prev === name ? null : name));
  }, []);

  const closeDropdown = useCallback(() => setOpenDropdown(null), []);

  /** Get display label for a chip based on active filter. */
  function chipLabel(type) {
    switch (type) {
      case 'intent':
        return filters.intent
          ? INTENT_OPTIONS.find((o) => o.value === filters.intent)?.label || filters.intent
          : 'Intent';
      case 'emotion':
        return filters.emotion
          ? EMOTION_OPTIONS.find((o) => o.value === filters.emotion)?.label || filters.emotion
          : 'Emotion';
      case 'entity':
        return filters.entity || 'Entity';
      case 'dates':
        if (filters.dateStart || filters.dateEnd) {
          const s = filters.dateStart ? filters.dateStart.slice(0, 10) : '…';
          const e = filters.dateEnd ? filters.dateEnd.slice(0, 10) : '…';
          return `${s} – ${e}`;
        }
        return 'Dates';
      default:
        return type;
    }
  }

  function isActive(type) {
    switch (type) {
      case 'intent': return !!filters.intent;
      case 'emotion': return !!filters.emotion;
      case 'entity': return !!filters.entity;
      case 'dates': return !!(filters.dateStart || filters.dateEnd);
      default: return false;
    }
  }

  function chipColor(type) {
    if (type === 'intent' && filters.intent) {
      return INTENT_OPTIONS.find((o) => o.value === filters.intent)?.color;
    }
    if (type === 'emotion' && filters.emotion) {
      return EMOTION_OPTIONS.find((o) => o.value === filters.emotion)?.color;
    }
    return null;
  }

  const CHIPS = [
    { id: 'intent', Icon: Target },
    { id: 'emotion', Icon: Heart },
    { id: 'entity', Icon: User },
    { id: 'dates', Icon: Calendar },
  ];

  return (
    <div className="filter-bar" role="toolbar" aria-label="Episode filters">
      {/* Filter icon with active count badge */}
      <div className="filter-bar__icon-wrapper">
        <Filter size={16} className="filter-bar__icon" aria-hidden="true" />
        {activeFilterCount > 0 && (
          <span className="filter-bar__badge" aria-label={`${activeFilterCount} active filters`}>
            {activeFilterCount}
          </span>
        )}
      </div>

      {/* Scrollable chip row */}
      <div className="filter-bar__chips">
        {CHIPS.map(({ id, Icon }) => (
          <div key={id} className="filter-chip-wrapper">
            <button
              className={`filter-chip ${isActive(id) ? 'filter-chip--active' : ''}`}
              onClick={() => toggle(id)}
              aria-expanded={openDropdown === id}
              aria-haspopup="listbox"
              style={chipColor(id) ? { '--chip-accent': chipColor(id) } : undefined}
            >
              <Icon size={14} className="filter-chip__icon" aria-hidden="true" />
              <span className="filter-chip__label">{chipLabel(id)}</span>
              <ChevronDown
                size={12}
                className={`filter-chip__arrow ${openDropdown === id ? 'filter-chip__arrow--open' : ''}`}
                aria-hidden="true"
              />
            </button>

            {/* Dropdown for intent */}
            {openDropdown === 'intent' && id === 'intent' && (
              <FilterDropdown
                options={INTENT_OPTIONS}
                value={filters.intent}
                onSelect={(v) => setFilter('intent', v)}
                onClose={closeDropdown}
                label="Filter by intent"
              />
            )}

            {/* Dropdown for emotion */}
            {openDropdown === 'emotion' && id === 'emotion' && (
              <FilterDropdown
                options={EMOTION_OPTIONS}
                value={filters.emotion}
                onSelect={(v) => setFilter('emotion', v)}
                onClose={closeDropdown}
                label="Filter by emotion"
              />
            )}

            {/* Entity text input */}
            {openDropdown === 'entity' && id === 'entity' && (
              <EntityInput
                value={filters.entity}
                onChange={(v) => setFilter('entity', v)}
                onClose={closeDropdown}
              />
            )}

            {/* Date range picker */}
            {openDropdown === 'dates' && id === 'dates' && (
              <DateRangeInput
                dateStart={filters.dateStart}
                dateEnd={filters.dateEnd}
                onChangeStart={(v) => setFilter('dateStart', v)}
                onChangeEnd={(v) => setFilter('dateEnd', v)}
                onClose={closeDropdown}
              />
            )}
          </div>
        ))}

        {/* Clear all button — only shown when filters are active */}
        {activeFilterCount > 0 && (
          <button
            className="filter-chip filter-chip--clear"
            onClick={() => { clearFilters(); closeDropdown(); }}
            aria-label="Clear all filters"
          >
            <X size={14} aria-hidden="true" />
            <span className="filter-chip__label">Clear</span>
          </button>
        )}
      </div>
    </div>
  );
}
