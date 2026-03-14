/**
 * AppShell — Root layout component wrapping all tab views.
 *
 * Provides:
 * - Premium BottomTabBar with React Router integration
 * - <Outlet /> for rendering the active tab's page component
 * - Persistent capture FAB (hidden on Chat tab, visible elsewhere)
 * - (Future) Shared WebSocket context
 *
 * Layout:
 * ┌──────────────────────┐
 * │                      │
 * │   <Outlet /> (page)  │
 * │                      │
 * │              [FAB] ──┤ (hidden on Chat)
 * ├──────────────────────┤
 * │  BottomTabBar        │
 * └──────────────────────┘
 */
import { Outlet, useLocation } from 'react-router-dom';
import { BottomTabBar } from './components/BottomTabBar';
import { CaptureFAB } from './components/CaptureFAB';
import { PageTransition } from './components/PageTransition';
import { CaptureProvider, useCapture } from './contexts/CaptureContext';
import './AppShell.css';

function AppShellInner() {
  const location = useLocation();
  const isChatTab = location.pathname === '/';
  const { captureBlurt } = useCapture();

  return (
    <div className="app-shell">
      <main className="app-shell__content">
        <PageTransition>
          <Outlet />
        </PageTransition>
      </main>
      <CaptureFAB visible={!isChatTab} onSubmit={captureBlurt} />
      <BottomTabBar />
    </div>
  );
}

export default function AppShell() {
  return (
    <CaptureProvider>
      <AppShellInner />
    </CaptureProvider>
  );
}
