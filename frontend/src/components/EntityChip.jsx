/**
 * EntityChip — compact tag showing an extracted entity from an episode.
 *
 * Displays entity name with an optional type indicator.
 */
import { User, MapPin, Building2, Tag } from 'lucide-react';

/** Map entity types to icons */
const TYPE_ICONS = {
  person: User,
  location: MapPin,
  place: MapPin,
  organization: Building2,
  company: Building2,
};

/**
 * @param {object} props
 * @param {string} props.name - entity name
 * @param {string} [props.entityType] - entity type (person, location, etc.)
 */
export function EntityChip({ name, entityType }) {
  if (!name) return null;

  const type = (entityType || '').toLowerCase();
  const IconComponent = TYPE_ICONS[type] || Tag;

  return (
    <span className="entity-chip" aria-label={`Entity: ${name}${type ? ` (${type})` : ''}`}>
      <IconComponent size={10} strokeWidth={2.5} aria-hidden="true" />
      <span>{name}</span>
    </span>
  );
}
