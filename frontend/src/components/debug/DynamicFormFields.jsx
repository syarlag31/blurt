/**
 * DynamicFormFields — Renders appropriate input controls based on
 * the selected endpoint's parameter definitions.
 *
 * Supports: text, number, json (textarea), toggle, select,
 * datetime, and hidden (auto-filled) field types.
 *
 * Props:
 *   fields    — Array of field definitions from endpointRegistry
 *   values    — Current form values object { fieldName: value }
 *   onChange  — (fieldName, value) => void
 *   errors    — Optional { fieldName: errorMessage } map
 */

import { useState, useCallback, useId } from 'react';
import { Lock } from 'lucide-react';
import './DynamicFormFields.css';

// ── Sub-components for each field type ──────────────────────────

function HiddenField({ field }) {
  return (
    <div className="dff__field">
      <div className="dff__label-row">
        <span className="dff__label">{field.label}</span>
        <span className="dff__location">{field.location}</span>
      </div>
      <div className="dff__hidden-chip">
        <Lock className="dff__hidden-icon" aria-hidden="true" />
        <span>{field.default ?? '—'}</span>
      </div>
    </div>
  );
}

function TextField({ field, value, onChange, error, inputId }) {
  return (
    <div className="dff__field">
      <FieldLabel field={field} htmlFor={inputId} />
      <input
        id={inputId}
        className={`dff__input${error ? ' dff__input--error' : ''}`}
        type="text"
        value={value ?? ''}
        onChange={(e) => onChange(field.name, e.target.value)}
        placeholder={field.placeholder || ''}
        required={field.required}
        autoComplete="off"
        autoCapitalize="off"
        spellCheck="false"
      />
      {error && <span className="dff__json-hint">{error}</span>}
    </div>
  );
}

function NumberField({ field, value, onChange, error, inputId }) {
  const rangeHints = [];
  if (field.min != null) rangeHints.push(`min: ${field.min}`);
  if (field.max != null) rangeHints.push(`max: ${field.max}`);
  if (field.step != null && field.step !== 1) rangeHints.push(`step: ${field.step}`);

  return (
    <div className="dff__field">
      <FieldLabel field={field} htmlFor={inputId} />
      <input
        id={inputId}
        className={`dff__input${error ? ' dff__input--error' : ''}`}
        type="number"
        value={value ?? ''}
        onChange={(e) => {
          const raw = e.target.value;
          // Allow empty string so the user can clear the field
          onChange(field.name, raw === '' ? '' : Number(raw));
        }}
        placeholder={field.placeholder || (field.default != null ? String(field.default) : '')}
        min={field.min}
        max={field.max}
        step={field.step ?? 'any'}
        inputMode="decimal"
        required={field.required}
        autoComplete="off"
      />
      {rangeHints.length > 0 && (
        <span className="dff__range-hint">{rangeHints.join(' · ')}</span>
      )}
      {error && <span className="dff__json-hint">{error}</span>}
    </div>
  );
}

function SelectField({ field, value, onChange, inputId }) {
  const options = field.options || [];

  return (
    <div className="dff__field">
      <FieldLabel field={field} htmlFor={inputId} />
      <div className="dff__select-wrap">
        <select
          id={inputId}
          className="dff__select"
          value={value ?? field.default ?? ''}
          onChange={(e) => onChange(field.name, e.target.value)}
          required={field.required}
        >
          {/* Show empty option if no default or if options include '' */}
          {!field.required && !options.includes('') && (
            <option value="">— select —</option>
          )}
          {options.map((opt) => (
            <option key={opt} value={opt}>
              {opt || '— none —'}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function JsonField({ field, value, onChange, inputId }) {
  const [jsonError, setJsonError] = useState(null);

  const handleChange = useCallback((e) => {
    const raw = e.target.value;
    onChange(field.name, raw);

    // Validate JSON as user types (debounced feel via try/catch)
    if (raw.trim() === '') {
      setJsonError(null);
      return;
    }
    try {
      JSON.parse(raw);
      setJsonError(null);
    } catch (err) {
      setJsonError('Invalid JSON');
    }
  }, [field.name, onChange]);

  return (
    <div className="dff__field">
      <FieldLabel field={field} htmlFor={inputId} />
      <textarea
        id={inputId}
        className={`dff__textarea${jsonError ? ' dff__textarea--invalid' : ''}`}
        value={value ?? ''}
        onChange={handleChange}
        placeholder={field.placeholder || '{}'}
        required={field.required}
        autoComplete="off"
        autoCapitalize="off"
        spellCheck="false"
        rows={3}
      />
      {jsonError && <span className="dff__json-hint">{jsonError}</span>}
    </div>
  );
}

function ToggleField({ field, value, onChange }) {
  const isOn = value === true || value === 'true';

  const handleToggle = useCallback(() => {
    onChange(field.name, !isOn);
  }, [field.name, isOn, onChange]);

  return (
    <div className="dff__toggle-row">
      <span className="dff__toggle-label">
        {field.label}
        {field.required && <span className="dff__toggle-required">*</span>}
        <span className="dff__toggle-location">{field.location}</span>
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={isOn}
        aria-label={field.label}
        className={`dff__toggle${isOn ? ' dff__toggle--on' : ''}`}
        onClick={handleToggle}
      >
        <span className="dff__toggle-thumb" />
      </button>
    </div>
  );
}

function DatetimeField({ field, value, onChange, inputId }) {
  return (
    <div className="dff__field">
      <FieldLabel field={field} htmlFor={inputId} />
      <input
        id={inputId}
        className="dff__input"
        type="datetime-local"
        value={value ?? ''}
        onChange={(e) => onChange(field.name, e.target.value)}
        required={field.required}
      />
    </div>
  );
}

// ── Shared label component ──────────────────────────────────────

function FieldLabel({ field, htmlFor }) {
  return (
    <div className="dff__label-row">
      <label className="dff__label" htmlFor={htmlFor}>
        {field.label}
        {field.required && <span className="dff__required"> *</span>}
      </label>
      <span className="dff__location">{field.location}</span>
    </div>
  );
}

// ── Main exported component ─────────────────────────────────────

export default function DynamicFormFields({ fields, values, onChange, errors }) {
  const idPrefix = useId();

  if (!fields || fields.length === 0) {
    return (
      <div className="dff dff--empty">
        <span className="dff__empty-text">
          No parameters — this endpoint takes no input
        </span>
      </div>
    );
  }

  return (
    <div className="dff">
      {fields.map((field) => {
        const inputId = `${idPrefix}-${field.name}`;
        const value = values[field.name];
        const error = errors?.[field.name];

        switch (field.type) {
          case 'hidden':
            return <HiddenField key={field.name} field={field} />;

          case 'text':
            return (
              <TextField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                error={error}
                inputId={inputId}
              />
            );

          case 'number':
            return (
              <NumberField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                error={error}
                inputId={inputId}
              />
            );

          case 'select':
            return (
              <SelectField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                inputId={inputId}
              />
            );

          case 'json':
            return (
              <JsonField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                inputId={inputId}
              />
            );

          case 'toggle':
            return (
              <ToggleField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
              />
            );

          case 'datetime':
            return (
              <DatetimeField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                inputId={inputId}
              />
            );

          default:
            return (
              <TextField
                key={field.name}
                field={field}
                value={value}
                onChange={onChange}
                error={error}
                inputId={inputId}
              />
            );
        }
      })}
    </div>
  );
}

// ── Utility: build initial values from field definitions ────────

/**
 * Creates a values object pre-filled with defaults from the endpoint fields.
 * Call this when the user selects a new endpoint.
 */
export function buildDefaultValues(fields) {
  const vals = {};
  if (!fields) return vals;
  for (const f of fields) {
    if (f.default != null) {
      vals[f.name] = f.default;
    }
  }
  return vals;
}

/**
 * Validates required fields. Returns { valid, errors }.
 */
export function validateFields(fields, values) {
  const errors = {};
  let valid = true;

  if (!fields) return { valid: true, errors };

  for (const f of fields) {
    if (f.type === 'hidden') continue; // auto-filled, always valid

    if (f.required) {
      const v = values[f.name];
      if (v == null || v === '') {
        errors[f.name] = `${f.label} is required`;
        valid = false;
      }
    }

    // Validate JSON fields
    if (f.type === 'json' && values[f.name]) {
      const raw = String(values[f.name]).trim();
      if (raw !== '') {
        try {
          JSON.parse(raw);
        } catch {
          errors[f.name] = 'Invalid JSON';
          valid = false;
        }
      }
    }
  }

  return { valid, errors };
}

/**
 * Builds the final request payload from field definitions and form values.
 * Separates into { pathParams, queryParams, body } based on field.location.
 */
export function buildRequestPayload(fields, values) {
  const pathParams = {};
  const queryParams = {};
  const body = {};

  if (!fields) return { pathParams, queryParams, body };

  for (const f of fields) {
    // Use default for hidden fields
    const val = f.type === 'hidden' ? f.default : values[f.name];

    // Skip empty optional values
    if (val == null || val === '') continue;

    // Parse JSON fields
    let finalVal = val;
    if (f.type === 'json' && typeof val === 'string') {
      try {
        finalVal = JSON.parse(val);
      } catch {
        finalVal = val; // let the server reject it
      }
    }

    // Convert toggle string to bool
    if (f.type === 'toggle') {
      finalVal = val === true || val === 'true';
    }

    switch (f.location) {
      case 'path':
        pathParams[f.name] = finalVal;
        break;
      case 'query':
        queryParams[f.name] = finalVal;
        break;
      case 'body':
        body[f.name] = finalVal;
        break;
    }
  }

  return { pathParams, queryParams, body };
}
