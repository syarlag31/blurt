/**
 * RequestPanel — Dynamic form fields, execute button, and response viewer
 * for the Debug tab API explorer.
 *
 * Receives a selected endpoint from the EndpointPicker, renders dynamic
 * form fields based on the endpoint's field definitions, builds the URL
 * and body, executes the request, and displays the response via JsonViewer.
 */
import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  Play,
  Loader2,
  Trash2,
  Clock,
  ChevronDown,
  Lock,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';
import JsonViewer from './JsonViewer';
import { ENDPOINT_CATEGORIES, ENDPOINTS_BY_ID, METHOD_COLORS } from './endpointRegistry';
import './RequestPanel.css';

/* ═══════════════════════════════════════════════════════════════════════
 * URL builder: replaces path params, appends query params
 * ═══════════════════════════════════════════════════════════════════════ */
function buildUrl(endpoint, fieldValues) {
  let url = endpoint.path;

  // Replace {param} in path
  for (const field of endpoint.fields) {
    if (field.location === 'path') {
      const val = fieldValues[field.name] ?? field.default ?? '';
      url = url.replace(`{${field.name}}`, encodeURIComponent(val));
    }
  }

  // Append query params
  const queryFields = endpoint.fields.filter((f) => f.location === 'query');
  const parts = [];
  for (const field of queryFields) {
    const val = fieldValues[field.name];
    if (val !== undefined && val !== '' && val !== null) {
      parts.push(`${field.name}=${encodeURIComponent(val)}`);
    }
  }
  if (parts.length) url += `?${parts.join('&')}`;

  return url;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Body builder: constructs JSON body from body-located fields
 * ═══════════════════════════════════════════════════════════════════════ */
function buildBody(endpoint, fieldValues) {
  const bodyFields = endpoint.fields.filter((f) => f.location === 'body');
  if (bodyFields.length === 0) return null;

  const body = {};
  for (const field of bodyFields) {
    const raw = fieldValues[field.name];
    if (raw === undefined || raw === '') continue;

    if (field.type === 'json') {
      try {
        body[field.name] = JSON.parse(raw);
      } catch {
        body[field.name] = raw; // send raw if invalid JSON
      }
    } else if (field.type === 'number') {
      const num = Number(raw);
      if (!isNaN(num)) body[field.name] = num;
    } else if (field.type === 'toggle') {
      body[field.name] = raw === true || raw === 'true';
    } else {
      body[field.name] = raw;
    }
  }

  return Object.keys(body).length > 0 ? JSON.stringify(body, null, 2) : null;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Field renderers
 * ═══════════════════════════════════════════════════════════════════════ */

function FieldText({ field, value, onChange }) {
  return (
    <input
      id={`field-${field.name}`}
      type="text"
      className="rp-field__input"
      placeholder={field.placeholder || ''}
      value={value ?? ''}
      onChange={(e) => onChange(field.name, e.target.value)}
      required={field.required}
    />
  );
}

function FieldNumber({ field, value, onChange }) {
  return (
    <input
      id={`field-${field.name}`}
      type="number"
      className="rp-field__input rp-field__input--number"
      placeholder={field.placeholder || ''}
      value={value ?? ''}
      min={field.min}
      max={field.max}
      step={field.step}
      onChange={(e) => onChange(field.name, e.target.value)}
      required={field.required}
    />
  );
}

function FieldSelect({ field, value, onChange }) {
  return (
    <div className="rp-field__select-wrap">
      <select
        id={`field-${field.name}`}
        className="rp-field__select"
        value={value ?? field.default ?? ''}
        onChange={(e) => onChange(field.name, e.target.value)}
        required={field.required}
      >
        {(!field.options?.includes('') && !field.default) && (
          <option value="">— select —</option>
        )}
        {field.options?.map((opt) => (
          <option key={opt} value={opt}>
            {opt || '(none)'}
          </option>
        ))}
      </select>
      <ChevronDown size={14} className="rp-field__select-chevron" />
    </div>
  );
}

function FieldJson({ field, value, onChange }) {
  return (
    <textarea
      id={`field-${field.name}`}
      className="rp-field__textarea"
      placeholder={field.placeholder || '{}'}
      value={value ?? ''}
      onChange={(e) => onChange(field.name, e.target.value)}
      rows={3}
      spellCheck={false}
      required={field.required}
    />
  );
}

function FieldDatetime({ field, value, onChange }) {
  return (
    <input
      id={`field-${field.name}`}
      type="datetime-local"
      className="rp-field__input rp-field__input--datetime"
      value={value ?? ''}
      onChange={(e) => onChange(field.name, e.target.value)}
      required={field.required}
    />
  );
}

function FieldToggle({ field, value, onChange }) {
  const isOn = value === true || value === 'true';
  return (
    <button
      type="button"
      className={`rp-field__toggle ${isOn ? 'rp-field__toggle--on' : ''}`}
      onClick={() => onChange(field.name, !isOn)}
      role="switch"
      aria-checked={isOn}
      aria-label={field.label}
    >
      {isOn ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
      <span className="rp-field__toggle-label">{isOn ? 'On' : 'Off'}</span>
    </button>
  );
}

function FieldHidden({ field, value }) {
  return (
    <div className="rp-field__hidden">
      <Lock size={12} />
      <span className="rp-field__hidden-value">{value ?? field.default}</span>
    </div>
  );
}

/** Render a single field with label */
function FormField({ field, value, onChange }) {
  const isHidden = field.type === 'hidden';
  let FieldComponent;

  switch (field.type) {
    case 'text': FieldComponent = FieldText; break;
    case 'number': FieldComponent = FieldNumber; break;
    case 'select': FieldComponent = FieldSelect; break;
    case 'json': FieldComponent = FieldJson; break;
    case 'datetime': FieldComponent = FieldDatetime; break;
    case 'toggle': FieldComponent = FieldToggle; break;
    case 'hidden': FieldComponent = FieldHidden; break;
    default: FieldComponent = FieldText;
  }

  return (
    <div className={`rp-field ${isHidden ? 'rp-field--hidden' : ''}`}>
      <label className="rp-field__label" htmlFor={`field-${field.name}`}>
        <span className="rp-field__label-text">{field.label}</span>
        {field.required && <span className="rp-field__required">*</span>}
        <span className={`rp-field__location rp-field__location--${field.location}`}>
          {field.location}
        </span>
      </label>
      <FieldComponent field={field} value={value} onChange={onChange} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════
 * RequestPanel — main component
 * ═══════════════════════════════════════════════════════════════════════ */
export default function RequestPanel({ endpoint, onClear }) {
  const [fieldValues, setFieldValues] = useState({});
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [requestMeta, setRequestMeta] = useState(null);
  const prevEndpointId = useRef(null);

  // Reset form when endpoint changes
  useEffect(() => {
    if (!endpoint || endpoint.id === prevEndpointId.current) return;
    prevEndpointId.current = endpoint.id;

    // Pre-fill defaults
    const defaults = {};
    for (const field of endpoint.fields) {
      if (field.default !== undefined) {
        defaults[field.name] = field.type === 'toggle' ? field.default : String(field.default);
      }
    }
    setFieldValues(defaults);
    setResponse(null);
    setError(null);
    setRequestMeta(null);
  }, [endpoint]);

  const handleFieldChange = useCallback((name, value) => {
    setFieldValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  /* ── Execute request ─────────────────────────────────────────────── */
  const handleExecute = useCallback(async () => {
    if (!endpoint) return;

    setLoading(true);
    setError(null);
    setResponse(null);

    const url = buildUrl(endpoint, fieldValues);
    const bodyStr = ['POST', 'PUT', 'PATCH'].includes(endpoint.method)
      ? buildBody(endpoint, fieldValues)
      : null;

    const startTime = performance.now();

    setRequestMeta({
      method: endpoint.method,
      url,
      body: bodyStr,
      timestamp: new Date().toLocaleTimeString(),
    });

    const fetchOptions = {
      method: endpoint.method,
      headers: {},
    };

    if (bodyStr) {
      fetchOptions.headers['Content-Type'] = 'application/json';
      fetchOptions.body = bodyStr;
    }

    try {
      const res = await fetch(url, fetchOptions);
      const elapsed = Math.round(performance.now() - startTime);
      const contentType = res.headers.get('content-type') || '';

      let data;
      if (contentType.includes('json')) {
        data = await res.json();
      } else {
        const text = await res.text();
        data = text ? { __raw_response: text } : null;
      }

      setResponse({ status: res.status, statusText: res.statusText, elapsed, data });
      if (!res.ok) {
        setError(`${res.status} ${res.statusText}`);
      }
    } catch (err) {
      const elapsed = Math.round(performance.now() - startTime);
      setError(err.message || 'Network error');
      setResponse({ status: 0, statusText: 'Error', elapsed, data: null });
    } finally {
      setLoading(false);
    }
  }, [endpoint, fieldValues]);

  const handleClearResponse = useCallback(() => {
    setResponse(null);
    setError(null);
    setRequestMeta(null);
  }, []);

  const handleRetry = useCallback(() => {
    handleExecute();
  }, [handleExecute]);

  // Status badge class
  const statusClass = response
    ? response.status >= 200 && response.status < 300
      ? 'success'
      : response.status >= 400
        ? 'error'
        : 'warn'
    : null;

  if (!endpoint) {
    return (
      <div className="rp rp--empty">
        <div className="rp__empty-state">
          <Play size={28} className="rp__empty-icon" />
          <p className="rp__empty-text">
            Select an endpoint above to start exploring the API.
          </p>
        </div>
      </div>
    );
  }

  // Split fields into groups for better visual hierarchy
  const hiddenFields = endpoint.fields.filter((f) => f.type === 'hidden');
  const visibleFields = endpoint.fields.filter((f) => f.type !== 'hidden');
  const pathFields = visibleFields.filter((f) => f.location === 'path');
  const queryFields = visibleFields.filter((f) => f.location === 'query');
  const bodyFields = visibleFields.filter((f) => f.location === 'body');

  return (
    <div className="rp">
      {/* ── Endpoint header ────────────────────────────────────────── */}
      <div className="rp__header">
        <span
          className="rp__method"
          style={{ '--method-color': METHOD_COLORS[endpoint.method] }}
        >
          {endpoint.method}
        </span>
        <div className="rp__endpoint-info">
          <span className="rp__endpoint-label">{endpoint.label}</span>
          <span className="rp__endpoint-path">{endpoint.path}</span>
        </div>
        {onClear && (
          <button className="rp__close" onClick={onClear} aria-label="Close endpoint">
            <Trash2 size={16} />
          </button>
        )}
      </div>

      {/* ── Hidden (auto-filled) params ────────────────────────────── */}
      {hiddenFields.length > 0 && (
        <div className="rp__hidden-fields">
          {hiddenFields.map((field) => (
            <div key={field.name} className="rp__hidden-chip">
              <Lock size={10} />
              <span className="rp__hidden-chip-name">{field.name}</span>
              <span className="rp__hidden-chip-value">{field.default}</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Path parameters ────────────────────────────────────────── */}
      {pathFields.length > 0 && (
        <div className="rp__field-group">
          <div className="rp__group-title">Path Parameters</div>
          {pathFields.map((field) => (
            <FormField
              key={field.name}
              field={field}
              value={fieldValues[field.name]}
              onChange={handleFieldChange}
            />
          ))}
        </div>
      )}

      {/* ── Query parameters ───────────────────────────────────────── */}
      {queryFields.length > 0 && (
        <div className="rp__field-group">
          <div className="rp__group-title">Query Parameters</div>
          {queryFields.map((field) => (
            <FormField
              key={field.name}
              field={field}
              value={fieldValues[field.name]}
              onChange={handleFieldChange}
            />
          ))}
        </div>
      )}

      {/* ── Body fields ────────────────────────────────────────────── */}
      {bodyFields.length > 0 && (
        <div className="rp__field-group">
          <div className="rp__group-title">Request Body</div>
          {bodyFields.map((field) => (
            <FormField
              key={field.name}
              field={field}
              value={fieldValues[field.name]}
              onChange={handleFieldChange}
            />
          ))}
        </div>
      )}

      {/* ── Execute bar ────────────────────────────────────────────── */}
      <div className="rp__actions">
        <button
          className="rp__execute"
          onClick={handleExecute}
          disabled={loading}
        >
          {loading ? (
            <>
              <Loader2 size={18} className="rp__spinner" />
              <span>Executing…</span>
            </>
          ) : (
            <>
              <Play size={18} />
              <span>Execute</span>
            </>
          )}
        </button>
        {response && (
          <button
            className="rp__clear-btn"
            onClick={handleClearResponse}
            aria-label="Clear response"
          >
            <Trash2 size={16} />
            <span>Clear</span>
          </button>
        )}
      </div>

      {/* ── Request summary ────────────────────────────────────────── */}
      {requestMeta && (
        <div className="rp__request-meta">
          <span
            className="rp__meta-method"
            style={{ '--method-color': METHOD_COLORS[requestMeta.method] }}
          >
            {requestMeta.method}
          </span>
          <span className="rp__meta-url">{requestMeta.url}</span>
          <span className="rp__meta-time">
            <Clock size={12} />
            {requestMeta.timestamp}
          </span>
        </div>
      )}

      {/* ── Status bar ─────────────────────────────────────────────── */}
      {response && (
        <div className={`rp__status-bar rp__status-bar--${statusClass}`}>
          <span className="rp__status-code">{response.status || 'ERR'}</span>
          <span className="rp__status-text">{response.statusText}</span>
          <span className="rp__status-elapsed">{response.elapsed}ms</span>
        </div>
      )}

      {/* ── Response viewer ────────────────────────────────────────── */}
      <div className="rp__response">
        <JsonViewer
          data={response?.data}
          title="Response"
          loading={loading}
          error={error && !response?.data ? error : null}
          onRetry={handleRetry}
          defaultExpanded={true}
          maxDepth={3}
        />
      </div>
    </div>
  );
}
