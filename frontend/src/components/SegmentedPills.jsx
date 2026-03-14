/**
 * SegmentedPills — A segmented control with pill-style active indicator.
 *
 * Used in the Memory tab to switch between sub-views:
 *   Timeline | Patterns | Rhythms | Graph
 *
 * Features:
 * - Animated sliding pill indicator
 * - Touch-friendly 44px+ targets
 * - Works with both dark and light themes
 * - Accessible with proper ARIA roles
 */
import { useRef, useState, useEffect, useCallback } from 'react';
import './SegmentedPills.css';

export function SegmentedPills({ segments, activeId, onChange }) {
  const containerRef = useRef(null);
  const [pillStyle, setPillStyle] = useState({});

  const updatePill = useCallback(() => {
    if (!containerRef.current) return;
    const activeBtn = containerRef.current.querySelector(
      `[data-segment-id="${activeId}"]`
    );
    if (!activeBtn) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const btnRect = activeBtn.getBoundingClientRect();

    setPillStyle({
      width: `${btnRect.width}px`,
      transform: `translateX(${btnRect.left - containerRect.left}px)`,
    });
  }, [activeId]);

  useEffect(() => {
    updatePill();
    // Re-measure on resize
    window.addEventListener('resize', updatePill);
    return () => window.removeEventListener('resize', updatePill);
  }, [updatePill]);

  return (
    <div
      className="seg-pills"
      ref={containerRef}
      role="tablist"
      aria-label="Memory sub-views"
    >
      {/* Sliding pill indicator */}
      <div
        className="seg-pills__indicator"
        style={pillStyle}
        aria-hidden="true"
      />

      {segments.map((seg) => (
        <button
          key={seg.id}
          data-segment-id={seg.id}
          className={`seg-pills__btn${
            activeId === seg.id ? ' seg-pills__btn--active' : ''
          }`}
          role="tab"
          aria-selected={activeId === seg.id}
          aria-controls={`panel-${seg.id}`}
          onClick={() => onChange(seg.id)}
        >
          {seg.icon && <seg.icon size={16} strokeWidth={2} aria-hidden="true" />}
          <span className="seg-pills__label">{seg.label}</span>
        </button>
      ))}
    </div>
  );
}
