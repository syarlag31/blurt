/**
 * EntityEditSheet — bottom sheet for entity operations.
 *
 * When a node is selected in the graph, this sheet slides up from the bottom
 * with three action buttons: Rename, Merge, Delete.
 *
 * Each action opens a ConfirmDialog with appropriate form fields.
 * After mutations, the graph state is updated via callbacks.
 */
import { useState, useCallback } from 'react';
import { useDrag } from '@use-gesture/react';
import {
  X,
  Pencil,
  GitMerge,
  Trash2,
  User,
  MapPin,
  Building2,
  Tag,
  Hash,
  Calendar,
} from 'lucide-react';
import { ConfirmDialog } from '../ConfirmDialog';
import './EntityEditSheet.css';

const TYPE_ICONS = {
  person: User,
  location: MapPin,
  place: MapPin,
  organization: Building2,
  company: Building2,
};

const DISMISS_THRESHOLD = 80;

/**
 * @param {Object} props
 * @param {Object|null} props.entity - Selected entity node
 * @param {Object[]} props.allNodes - All nodes for merge target picker
 * @param {(entityId: string, newName: string) => Promise<boolean>} props.onRename
 * @param {(keepId: string, removeId: string) => Promise<boolean>} props.onMerge
 * @param {(entityId: string) => Promise<boolean>} props.onDelete
 * @param {() => void} props.onClose
 * @param {boolean} [props.mutating] - Whether a mutation is in progress
 */
export function EntityEditSheet({
  entity,
  allNodes = [],
  onRename,
  onMerge,
  onDelete,
  onClose,
  mutating = false,
}) {
  const [dragY, setDragY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  // Dialog states
  const [renameOpen, setRenameOpen] = useState(false);
  const [mergeOpen, setMergeOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [renameName, setRenameName] = useState('');
  const [mergeTarget, setMergeTarget] = useState('');
  const [actionLoading, setActionLoading] = useState(false);

  const open = !!entity;

  // Swipe-to-dismiss gesture
  const bind = useDrag(
    ({ down, movement: [, my], cancel }) => {
      if (my < 0) {
        cancel();
        return;
      }
      setIsDragging(down);
      if (down) {
        setDragY(my);
      } else {
        if (my > DISMISS_THRESHOLD) {
          onClose();
        }
        setDragY(0);
      }
    },
    { axis: 'y', filterTaps: true }
  );

  // ── Rename ────────────────────────────────────────────
  const openRename = useCallback(() => {
    setRenameName(entity?.name || '');
    setRenameOpen(true);
  }, [entity]);

  const handleRename = useCallback(async () => {
    if (!entity || !renameName.trim()) return;
    setActionLoading(true);
    const ok = await onRename(entity.id, renameName.trim());
    setActionLoading(false);
    if (ok) {
      setRenameOpen(false);
      onClose();
    }
  }, [entity, renameName, onRename, onClose]);

  // ── Merge ─────────────────────────────────────────────
  const openMerge = useCallback(() => {
    setMergeTarget('');
    setMergeOpen(true);
  }, []);

  const handleMerge = useCallback(async () => {
    if (!entity || !mergeTarget) return;
    setActionLoading(true);
    const ok = await onMerge(entity.id, mergeTarget);
    setActionLoading(false);
    if (ok) {
      setMergeOpen(false);
      onClose();
    }
  }, [entity, mergeTarget, onMerge, onClose]);

  // ── Delete ────────────────────────────────────────────
  const openDelete = useCallback(() => {
    setDeleteOpen(true);
  }, []);

  const handleDelete = useCallback(async () => {
    if (!entity) return;
    setActionLoading(true);
    const ok = await onDelete(entity.id);
    setActionLoading(false);
    if (ok) {
      setDeleteOpen(false);
      onClose();
    }
  }, [entity, onDelete, onClose]);

  if (!open) return null;

  const IconComponent = TYPE_ICONS[entity.type] || Tag;
  const mergeTargets = allNodes.filter((n) => n.id !== entity.id);

  const sheetStyle = isDragging
    ? { transform: `translateY(${dragY}px)`, transition: 'none' }
    : { transform: 'translateY(0)', transition: 'transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94)' };

  return (
    <>
      {/* Backdrop */}
      <div
        className={`entity-sheet__backdrop ${open ? 'entity-sheet__backdrop--open' : ''}`}
        onClick={onClose}
        role="presentation"
      />

      {/* Sheet */}
      <div
        className="entity-sheet"
        style={sheetStyle}
        role="dialog"
        aria-modal="true"
        aria-label={`Edit entity: ${entity.name}`}
      >
        {/* Drag handle */}
        <div className="entity-sheet__handle-wrap" {...bind()}>
          <div className="entity-sheet__handle" />
        </div>

        {/* Header */}
        <div className="entity-sheet__header">
          <div className="entity-sheet__entity-info">
            <div className={`entity-sheet__type-icon entity-sheet__type-icon--${entity.type}`}>
              <IconComponent size={20} strokeWidth={2} />
            </div>
            <div>
              <h3 className="entity-sheet__name">{entity.name}</h3>
              <span className="entity-sheet__type-label">{entity.type || 'entity'}</span>
            </div>
          </div>
          <button
            className="entity-sheet__close"
            onClick={onClose}
            aria-label="Close"
          >
            <X size={20} />
          </button>
        </div>

        {/* Stats */}
        <div className="entity-sheet__stats">
          <div className="entity-sheet__stat">
            <Hash size={14} strokeWidth={2} />
            <span>{entity.mentions || 0} mention{entity.mentions !== 1 ? 's' : ''}</span>
          </div>
          {entity.firstSeen && (
            <div className="entity-sheet__stat">
              <Calendar size={14} strokeWidth={2} />
              <span>Since {new Date(entity.firstSeen).toLocaleDateString()}</span>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="entity-sheet__actions">
          <button
            className="entity-sheet__action entity-sheet__action--rename"
            onClick={openRename}
            disabled={mutating}
          >
            <Pencil size={18} strokeWidth={2} />
            <span>Rename</span>
          </button>

          <button
            className="entity-sheet__action entity-sheet__action--merge"
            onClick={openMerge}
            disabled={mutating || mergeTargets.length === 0}
          >
            <GitMerge size={18} strokeWidth={2} />
            <span>Merge</span>
          </button>

          <button
            className="entity-sheet__action entity-sheet__action--delete"
            onClick={openDelete}
            disabled={mutating}
          >
            <Trash2 size={18} strokeWidth={2} />
            <span>Delete</span>
          </button>
        </div>
      </div>

      {/* ── Rename Dialog ─────────────────────────────────── */}
      <ConfirmDialog
        open={renameOpen}
        title="Rename Entity"
        description={`Rename "${entity.name}" to a new name. This updates the entity across the graph.`}
        variant="rename"
        confirmLabel="Rename"
        loading={actionLoading}
        confirmDisabled={!renameName.trim() || renameName.trim().toLowerCase() === entity.id}
        onConfirm={handleRename}
        onCancel={() => setRenameOpen(false)}
      >
        <input
          type="text"
          className="entity-sheet__input"
          value={renameName}
          onChange={(e) => setRenameName(e.target.value)}
          placeholder="New entity name"
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Enter' && renameName.trim()) handleRename();
          }}
        />
      </ConfirmDialog>

      {/* ── Merge Dialog ──────────────────────────────────── */}
      <ConfirmDialog
        open={mergeOpen}
        title="Merge Entities"
        description={`Merge another entity into "${entity.name}". The selected entity will be absorbed — its connections and mentions transfer to "${entity.name}".`}
        variant="merge"
        confirmLabel="Merge"
        loading={actionLoading}
        confirmDisabled={!mergeTarget}
        onConfirm={handleMerge}
        onCancel={() => setMergeOpen(false)}
      >
        <div className="entity-sheet__merge-picker">
          <label className="entity-sheet__merge-label">Absorb entity:</label>
          <select
            className="entity-sheet__select"
            value={mergeTarget}
            onChange={(e) => setMergeTarget(e.target.value)}
          >
            <option value="">Select entity to merge...</option>
            {mergeTargets
              .sort((a, b) => a.name.localeCompare(b.name))
              .map((n) => (
                <option key={n.id} value={n.id}>
                  {n.name} ({n.mentions} mention{n.mentions !== 1 ? 's' : ''})
                </option>
              ))}
          </select>
        </div>
      </ConfirmDialog>

      {/* ── Delete Dialog ─────────────────────────────────── */}
      <ConfirmDialog
        open={deleteOpen}
        title="Delete Entity"
        description={`Permanently remove "${entity.name}" from the graph. This will also remove all ${(entity.mentions || 0)} connection${entity.mentions !== 1 ? 's' : ''}.`}
        variant="delete"
        confirmLabel="Delete"
        loading={actionLoading}
        onConfirm={handleDelete}
        onCancel={() => setDeleteOpen(false)}
      />
    </>
  );
}
