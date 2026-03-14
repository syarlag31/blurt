/**
 * Router configuration — defines all tab routes for the Blurt app.
 *
 * 4-tab bottom nav layout:
 *   /        → Chat (default, index route)
 *   /memory  → Memory (knowledge graph)
 *   /tasks   → Tasks (management + surfacing)
 *   /debug   → Debug (raw API + diagnostics)
 *
 * All routes share the same root layout (AppShell) which provides
 * the bottom navigation bar, persistent capture FAB, and shared
 * WebSocket context.
 */
import { createBrowserRouter, Navigate } from 'react-router-dom';
import AppShell from './AppShell';
import ChatPage from './pages/ChatPage';
import MemoryPage from './pages/MemoryPage';
import TasksPage from './pages/TasksPage';
import DebugPage from './pages/DebugPage';

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppShell />,
    children: [
      {
        index: true,
        element: <ChatPage />,
      },
      {
        path: 'memory',
        element: <MemoryPage />,
      },
      {
        path: 'tasks',
        element: <TasksPage />,
      },
      {
        path: 'debug',
        element: <DebugPage />,
      },
      {
        // Catch-all: redirect unknown routes back to chat
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);

export default router;
