/**
 * PageTransition — Animated wrapper for tab content transitions.
 *
 * Provides smooth fade + slide animations (200ms) when switching between tabs.
 * Direction is determined by tab index (left-to-right or right-to-left).
 * Uses CSS animations with transform + opacity for GPU-accelerated performance.
 */
import { useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import './PageTransition.css';

const TAB_ORDER = ['/', '/memory', '/tasks', '/debug'];

function getTabIndex(pathname) {
  const idx = TAB_ORDER.indexOf(pathname);
  return idx >= 0 ? idx : 0;
}

export function PageTransition({ children }) {
  const location = useLocation();
  const prevIndexRef = useRef(getTabIndex(location.pathname));
  const currentIndex = getTabIndex(location.pathname);

  const direction = currentIndex > prevIndexRef.current ? 'right' : 'left';

  // Update prev after determining direction
  const prevRef = useRef(prevIndexRef.current);
  if (prevRef.current !== currentIndex) {
    prevIndexRef.current = prevRef.current;
    prevRef.current = currentIndex;
  }

  // Choose animation class based on direction
  const animClass =
    currentIndex === prevIndexRef.current
      ? 'page-transition--enter'
      : direction === 'right'
        ? 'page-transition--slide-from-right'
        : 'page-transition--slide-from-left';

  return (
    <div
      key={location.pathname}
      className={`page-transition ${animClass}`}
    >
      {children}
    </div>
  );
}
