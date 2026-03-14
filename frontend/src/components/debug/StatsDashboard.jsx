/**
 * StatsDashboard — Comprehensive stats grid for the Debug tab.
 *
 * Fetches from all backend health/stats endpoints and displays:
 *   1. Server Health     — GET /health
 *   2. Classification    — GET /api/v1/classify/health + /stats
 *   3. Capture Pipeline  — GET /api/v1/blurt/stats
 *   4. Recall Engine     — GET /api/v1/recall/stats
 *   5. Google Calendar   — GET /auth/google/status
 *
 * Features:
 *   - Loading skeleton state per card (shimmer animation)
 *   - Error states with retry tap
 *   - Real-time updates via refreshKey prop (WebSocket blurt.created)
 *   - 30s polling interval for background freshness
 *   - Value change animation (pop effect)
 */
import {
  Activity,
  Database,
  Brain,
  Calendar,
  Server,
  RefreshCw,
  AlertCircle,
} from 'lucide-react';
import { StatsCard } from '../StatsCard';
import { useStatsData } from '../../hooks/useStatsData';
import './StatsDashboard.css';

/* ── Stat derivation helpers ─────────────────────────────── */

function deriveHealth(cat) {
  if (cat.loading) return { loading: true };
  if (cat.error) return errorCard('Server', cat.error);
  const d = cat.data;
  if (!d) return errorCard('Server', 'No data');
  return {
    label: 'Server',
    value: d.status === 'ok' ? 'Online' : d.status || '—',
    status: d.status === 'ok' ? 'active' : 'warning',
    variant: d.status === 'ok' ? 'success' : 'warning',
    trend: d.active_sessions != null
      ? { direction: 'neutral', label: `${d.active_sessions} sessions` }
      : undefined,
  };
}

function deriveClassify(healthCat, statsCat) {
  if (healthCat.loading || statsCat.loading) return { loading: true };
  if (healthCat.error && statsCat.error) return errorCard('Classify', healthCat.error);
  const h = healthCat.data;
  const s = statsCat.data;
  if (!h && !s) return errorCard('Classify', 'No data');

  const healthy = h?.healthy ?? true;
  const total = s?.total_classified ?? h?.total_classified ?? 0;
  const latency = s?.avg_latency_ms ?? h?.avg_latency_ms;
  const confRate = s?.confident_rate;

  return {
    label: 'Classify',
    value: total,
    status: healthy ? 'active' : 'error',
    variant: healthy ? 'success' : 'danger',
    trend: latency != null
      ? { direction: latency < 200 ? 'up' : 'down', label: `${Math.round(latency)}ms` }
      : confRate != null
        ? { direction: confRate > 0.85 ? 'up' : 'down', label: `${(confRate * 100).toFixed(0)}%` }
        : undefined,
  };
}

function deriveCapture(cat) {
  if (cat.loading) return { loading: true };
  if (cat.error) return errorCard('Captures', cat.error);
  const d = cat.data;
  if (!d) return errorCard('Captures', 'No data');

  const total = d.total_captured ?? 0;
  const dropRate = d.drop_rate ?? 0;
  const voice = d.voice_count ?? 0;
  const text = d.text_count ?? 0;

  return {
    label: 'Captures',
    value: total,
    status: dropRate === 0 ? 'active' : 'warning',
    variant: dropRate === 0 ? 'info' : 'warning',
    trend: total > 0
      ? { direction: 'neutral', label: `${voice}v / ${text}t` }
      : undefined,
  };
}

function deriveRecall(cat) {
  if (cat.loading) return { loading: true };
  if (cat.error) return errorCard('Recall', cat.error);
  const d = cat.data;
  if (!d) return errorCard('Recall', 'No data');

  const queries = d.total_queries ?? 0;
  const latency = d.avg_latency_ms;
  const results = d.total_results_returned ?? 0;

  return {
    label: 'Recall',
    value: queries,
    status: queries > 0 ? 'active' : 'inactive',
    variant: 'info',
    trend: latency != null
      ? { direction: latency < 500 ? 'up' : 'down', label: `${Math.round(latency)}ms avg` }
      : results > 0
        ? { direction: 'neutral', label: `${results} results` }
        : undefined,
  };
}

function deriveCalendar(cat) {
  if (cat.loading) return { loading: true };
  if (cat.error) return errorCard('Calendar', cat.error);
  const d = cat.data;
  if (!d) return errorCard('Calendar', 'No data');

  const connected = d.connected ?? d.is_connected ?? false;
  return {
    label: 'Calendar',
    value: connected ? 'Connected' : 'Off',
    status: connected ? 'active' : 'inactive',
    variant: connected ? 'success' : 'warning',
    trend: d.email ? { direction: 'neutral', label: d.email } : undefined,
  };
}

/** Error card helper. */
function errorCard(label, errorMsg) {
  return {
    label,
    value: 'Error',
    status: 'error',
    variant: 'danger',
    trend: { direction: 'down', label: errorMsg || 'Failed' },
  };
}

/* ── Component ───────────────────────────────────────────── */

export default function StatsDashboard({ refreshKey }) {
  const { stats, refetch } = useStatsData(refreshKey);

  const health = deriveHealth(stats.health);
  const classify = deriveClassify(stats.classify, stats.classifyStats);
  const capture = deriveCapture(stats.capture);
  const recall = deriveRecall(stats.recall);
  const calendar = deriveCalendar(stats.calendar);

  const anyLoading = Object.values(stats).some((c) => c.loading);
  const lastUpdate = Math.max(
    ...Object.values(stats).map((c) => c.lastFetched || 0),
  );

  return (
    <section className="stats-dashboard" aria-label="System stats">
      <div className="stats-dashboard__header">
        <h3 className="stats-dashboard__title">System Health</h3>
        <button
          className="stats-dashboard__refresh"
          onClick={refetch}
          aria-label="Refresh stats"
          disabled={anyLoading}
        >
          <RefreshCw
            size={16}
            className={anyLoading ? 'stats-dashboard__spin' : ''}
          />
          {lastUpdate > 0 && (
            <span className="stats-dashboard__timestamp">
              {formatAge(lastUpdate)}
            </span>
          )}
        </button>
      </div>

      <div className="stats-dashboard__grid">
        <StatsCard
          icon={Server}
          onClick={refetch}
          {...health}
        />
        <StatsCard
          icon={Activity}
          onClick={refetch}
          {...classify}
        />
        <StatsCard
          icon={Database}
          onClick={refetch}
          {...capture}
        />
        <StatsCard
          icon={Brain}
          onClick={refetch}
          {...recall}
        />
        <StatsCard
          icon={Calendar}
          onClick={refetch}
          {...calendar}
        />
      </div>

      {/* Per-category error summary */}
      <ErrorSummary stats={stats} onRetry={refetch} />
    </section>
  );
}

/* ── Error summary banner ────────────────────────────────── */

function ErrorSummary({ stats, onRetry }) {
  const errors = Object.entries(stats)
    .filter(([, cat]) => cat.error)
    .map(([key, cat]) => ({ key, error: cat.error }));

  if (errors.length === 0) return null;

  return (
    <div className="stats-dashboard__errors" role="alert">
      <div className="stats-dashboard__errors-header">
        <AlertCircle size={14} />
        <span>
          {errors.length} endpoint{errors.length > 1 ? 's' : ''} unreachable
        </span>
      </div>
      <ul className="stats-dashboard__errors-list">
        {errors.map(({ key, error }) => (
          <li key={key}>
            <strong>{key}:</strong> {error}
          </li>
        ))}
      </ul>
      <button className="stats-dashboard__retry-btn" onClick={onRetry}>
        <RefreshCw size={12} /> Retry all
      </button>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────────────── */

function formatAge(timestamp) {
  const seconds = Math.round((Date.now() - timestamp) / 1000);
  if (seconds < 5) return 'just now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.round(seconds / 60);
  return `${minutes}m ago`;
}
