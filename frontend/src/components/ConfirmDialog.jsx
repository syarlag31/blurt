/**
 * ConfirmDialog — accessible confirmation dialog with backdrop.
 *
 * Renders a centered modal with title, description, and cancel/confirm buttons.
 * Supports destructive (red) and default (blue) variants.
 * Closes on backdrop tap, Escape key, or cancel button.
 */
import { useCallback, useEffect, useRef } from 'react';
import { AlertTriangle, Trash2, GitMerge, Pencil } from 'lucide-react';
import './ConfirmDialog.css';

const VARIANT_ICONS = {
  delete: Trash2,
  merge: GitMerge,
  rename: Pencil,
  warning: AlertTriangle,
};

/**
 * @param {Object} props
 * @param {boolean} props.open - Whether dialog is visible
 * @param {string} props.title - Dialog title
 * @param {string} [props.description] - Explanatory text
 * @param {React.ReactNode} [props.children] - Additional content (e.g., inputs)
 * @param {string} [props.confirmLabel='Confirm'] - Confirm button text
 * @param {string} [props.cancelLabel='Cancel'] - Cancel button text
 * @param {'delete'|'merge'|'rename'|'warning'} [props.variant='warning'] - Visual variant
 * @param {boolean} [props.loading] - Show loading state on confirm button
 * @param {boolean} [props.confirmDisabled] - Disable confirm button
 * @param {() => void} props.onConfirm - Called on confirm
 * @param {() => void} props.onCancel - Called on cancel/dismiss
 */
export function ConfirmDialog({
  open,
  title,
  description,
  children,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'warning',
  loading = false,
  confirmDisabled = false,
  onConfirm,
  onCancel,
}) {
  const dialogRef = useRef(null);
  const confirmBtnRef = useRef(null);

  // Trap focus and handle Escape
  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onCancel();
      }
    };
    document.addEventListener('keydown', handleKeyDown);

    // Prevent body scroll
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onCancel]);

  const handleBackdropClick = useCallback((e) => {
    if (e.target === e.currentTarget) {
      onCancel();
    }
  }, [onCancel]);

  if (!open) return null;

  const isDestructive = variant === 'delete';
  const IconComponent = VARIANT_ICONS[variant] || AlertTriangle;

  return (
    <div
      className="confirm-dialog__backdrop"
      onClick={handleBackdropClick}
      role="presentation"
    >
      <div
        ref={dialogRef}
        className={`confirm-dialog ${isDestructive ? 'confirm-dialog--destructive' : ''}`}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirm-dialog-title"
        aria-describedby={description ? 'confirm-dialog-desc' : undefined}
      >
        <div className={`confirm-dialog__icon confirm-dialog__icon--${variant}`}>
          <IconComponent size={24} strokeWidth={2} />
        </div>

        <h2 id="confirm-dialog-title" className="confirm-dialog__title">
          {title}
        </h2>

        {description && (
          <p id="confirm-dialog-desc" className="confirm-dialog__description">
            {description}
          </p>
        )}

        {children && (
          <div className="confirm-dialog__content">
            {children}
          </div>
        )}

        <div className="confirm-dialog__actions">
          <button
            type="button"
            className="confirm-dialog__btn confirm-dialog__btn--cancel"
            onClick={onCancel}
            disabled={loading}
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmBtnRef}
            type="button"
            className={`confirm-dialog__btn confirm-dialog__btn--confirm ${
              isDestructive ? 'confirm-dialog__btn--danger' : ''
            }`}
            onClick={onConfirm}
            disabled={loading || confirmDisabled}
          >
            {loading ? (
              <span className="confirm-dialog__spinner" aria-hidden="true" />
            ) : null}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
