import { useState, useRef, useEffect, useCallback } from 'react'
import './SegmentedPill.css'

const SEGMENTS = ['Timeline', 'Patterns', 'Rhythms', 'Graph']

/**
 * SegmentedPill — A reusable horizontal pill-style tab selector.
 *
 * @param {Object} props
 * @param {string[]}  [props.options]   — Labels for each segment (default: Timeline/Patterns/Rhythms/Graph)
 * @param {string}    [props.value]     — Currently active segment label
 * @param {function}  props.onChange     — Callback with the selected segment label
 * @param {string}    [props.className] — Optional extra class names
 */
export default function SegmentedPill({
  options = SEGMENTS,
  value,
  onChange,
  className = '',
}) {
  const activeIndex = Math.max(0, options.indexOf(value))
  const containerRef = useRef(null)
  const [indicator, setIndicator] = useState({ left: 0, width: 0 })

  // Measure the active button to position the sliding indicator
  const updateIndicator = useCallback(() => {
    const container = containerRef.current
    if (!container) return
    const buttons = container.querySelectorAll('.seg-pill__btn')
    const btn = buttons[activeIndex]
    if (!btn) return
    const containerRect = container.getBoundingClientRect()
    const btnRect = btn.getBoundingClientRect()
    setIndicator({
      left: btnRect.left - containerRect.left,
      width: btnRect.width,
    })
  }, [activeIndex])

  useEffect(() => {
    updateIndicator()
    // Recalculate on resize (e.g. orientation change)
    window.addEventListener('resize', updateIndicator)
    return () => window.removeEventListener('resize', updateIndicator)
  }, [updateIndicator])

  return (
    <div
      className={`seg-pill ${className}`}
      ref={containerRef}
      role="tablist"
      aria-label="View selector"
    >
      {/* Animated sliding indicator */}
      <div
        className="seg-pill__indicator"
        style={{
          transform: `translateX(${indicator.left}px)`,
          width: `${indicator.width}px`,
        }}
        aria-hidden="true"
      />

      {options.map((label, i) => {
        const isActive = i === activeIndex
        return (
          <button
            key={label}
            className={`seg-pill__btn ${isActive ? 'seg-pill__btn--active' : ''}`}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange?.(label)}
            type="button"
          >
            {label}
          </button>
        )
      })}
    </div>
  )
}
