/**
 * DebugPage — Raw API access and system diagnostics.
 *
 * Layout (scrollable, mobile-first):
 *   1. System Health — 2x2 stats dashboard (top)
 *   2. API Explorer  — Endpoint picker accordion
 *   3. Request Panel — Dynamic form fields, execute, response viewer
 *
 * Exposes ALL backend REST endpoints via collapsible endpoint picker,
 * renders typed form fields, executes requests, and displays JSON responses.
 *
 * Stats refresh automatically via:
 *   - 30s polling interval
 *   - BroadcastChannel blurt.created events from Chat tab
 *   - Manual tap-to-refresh on any card
 */

import { useState, useCallback } from 'react';
import EndpointPicker from '../components/debug/EndpointPicker.jsx';
import StatsDashboard from '../components/debug/StatsDashboard.jsx';
import ActivityHeatmap from '../components/debug/ActivityHeatmap.jsx';
import RequestPanel from '../components/debug/RequestPanel.jsx';
import { ENDPOINTS_BY_ID } from '../components/debug/endpointRegistry.js';
import { useBroadcastRefresh } from '../hooks/useBroadcastRefresh';
import './DebugPage.css';

export default function DebugPage() {
  const [selectedEndpoint, setSelectedEndpoint] = useState(null);
  const refreshKey = useBroadcastRefresh();

  /** Resolve endpoint: match the picker's simple entry to the detailed registry with fields */
  const handleSelectEndpoint = useCallback((endpoint) => {
    // First try by id (if the picker entry has one)
    if (endpoint.id && ENDPOINTS_BY_ID[endpoint.id]) {
      setSelectedEndpoint(ENDPOINTS_BY_ID[endpoint.id]);
      return;
    }
    // Fallback: match by method + path
    const match = Object.values(ENDPOINTS_BY_ID).find(
      (ep) => ep.method === endpoint.method && ep.path === endpoint.path
    );
    setSelectedEndpoint(match || endpoint);
  }, []);

  const handleClearEndpoint = useCallback(() => {
    setSelectedEndpoint(null);
  }, []);

  return (
    <div className="page page--debug debug-page">
      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="debug-page__header">
        <h2 className="debug-page__title">Debug</h2>
        <p className="debug-page__subtitle">Raw API access & system diagnostics</p>
      </header>

      {/* ── System Health ──────────────────────────────────────────── */}
      <StatsDashboard refreshKey={refreshKey} />

      {/* ── Weekly Activity Heatmap ────────────────────────────────── */}
      <ActivityHeatmap refreshKey={refreshKey} />

      {/* ── API Explorer (endpoint picker accordion) ───────────────── */}
      <section className="debug-page__section" aria-label="API Explorer">
        <EndpointPicker
          onSelect={handleSelectEndpoint}
          selectedEndpoint={selectedEndpoint}
        />
      </section>

      {/* ── Request / Response Panel ───────────────────────────────── */}
      <section className="debug-page__section" aria-label="Request panel">
        <RequestPanel
          endpoint={selectedEndpoint}
          onClear={selectedEndpoint ? handleClearEndpoint : null}
        />
      </section>
    </div>
  );
}
