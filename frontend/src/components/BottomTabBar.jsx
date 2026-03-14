/**
 * BottomTabBar — 4-tab bottom navigation for Blurt dogfooding UI.
 *
 * Tabs: Chat, Memory, Tasks, Debug
 * - Lucide SVG icons (no emoji)
 * - Active state with accent glow + filled icon variant
 * - 44px+ touch targets
 * - Safe area inset for notched phones
 * - Premium dark theme with blue-tinted surfaces
 * - Micro-interaction on tap (scale spring)
 */
import { useCallback, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  MessageCircle,
  Brain,
  CheckSquare,
  Terminal,
} from 'lucide-react';
import './BottomTabBar.css';

const TABS = [
  { path: '/', key: 'chat', label: 'Chat', Icon: MessageCircle },
  { path: '/memory', key: 'memory', label: 'Memory', Icon: Brain },
  { path: '/tasks', key: 'tasks', label: 'Tasks', Icon: CheckSquare },
  { path: '/debug', key: 'debug', label: 'Debug', Icon: Terminal },
];

export function BottomTabBar() {
  const location = useLocation();
  const navigate = useNavigate();
  const activeRefs = useRef({});

  const isActive = useCallback(
    (path) => {
      if (path === '/') return location.pathname === '/';
      return location.pathname.startsWith(path);
    },
    [location.pathname],
  );

  const handleTabTap = useCallback(
    (path, key) => {
      // Micro-interaction: spring scale on tap
      const el = activeRefs.current[key];
      if (el) {
        el.classList.remove('tab-btn--pop');
        // Force reflow to restart animation
        void el.offsetWidth;
        el.classList.add('tab-btn--pop');
      }
      navigate(path);
    },
    [navigate],
  );

  return (
    <nav className="bottom-tab-bar" role="tablist" aria-label="Main navigation">
      {TABS.map(({ path, key, label, Icon }) => {
        const active = isActive(path);
        return (
          <button
            key={key}
            ref={(el) => {
              activeRefs.current[key] = el;
            }}
            role="tab"
            aria-selected={active}
            aria-label={label}
            className={`tab-btn ${active ? 'tab-btn--active' : ''}`}
            onClick={() => handleTabTap(path, key)}
          >
            {active && <span className="tab-btn__glow" aria-hidden="true" />}
            <span className="tab-btn__icon">
              <Icon
                size={22}
                strokeWidth={active ? 2.5 : 1.8}
                className={`tab-btn__svg ${active ? 'tab-btn__svg--active' : ''}`}
              />
            </span>
            <span className={`tab-btn__label ${active ? 'tab-btn__label--active' : ''}`}>
              {label}
            </span>
            {active && <span className="tab-btn__indicator" aria-hidden="true" />}
          </button>
        );
      })}
    </nav>
  );
}
