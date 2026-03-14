/**
 * GraphView — d3-force knowledge graph visualization with tap-to-explore.
 *
 * Renders an interactive force-directed graph of entities extracted
 * from episodes. Tapping a node highlights it, dims unconnected nodes,
 * shows connected edges, and opens an entity detail panel.
 *
 * Data sourced from episodes (entities array) — no new backend endpoints.
 */
import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
  forceX,
  forceY,
} from 'd3-force';
import { Share2, X, Clock, Hash, ArrowRight, Loader, Pencil } from 'lucide-react';
import { EntityEditSheet } from './EntityEditSheet';
import './GraphView.css';

/** Entity type → color mapping */
const TYPE_COLORS = {
  PERSON: '#6390ff',
  PLACE: '#10b981',
  ORGANIZATION: '#8b5cf6',
  PROJECT: '#f59e0b',
  TOOL: '#ec4899',
  CONCEPT: '#14b8a6',
  EVENT: '#f472b6',
  DEFAULT: '#94a3b8',
};

function getTypeColor(type) {
  if (!type) return TYPE_COLORS.DEFAULT;
  const upper = type.toUpperCase();
  return TYPE_COLORS[upper] || TYPE_COLORS.DEFAULT;
}

/** Build graph data from episodes: extract entities and co-occurrence edges */
function buildGraphFromEpisodes(episodes) {
  const entityMap = new Map(); // name -> { name, type, count, episodes[] }
  const edgeMap = new Map(); // "a|b" -> { source, target, weight }

  for (const ep of episodes) {
    const entities = ep.entities || [];
    const names = [];

    for (const ent of entities) {
      const name = typeof ent === 'string' ? ent : ent.name || ent.label;
      const type = typeof ent === 'string' ? null : ent.type || ent.entity_type;
      if (!name) continue;

      const key = name.toLowerCase();
      if (!entityMap.has(key)) {
        entityMap.set(key, {
          id: key,
          name,
          type: type || 'DEFAULT',
          count: 0,
          episodeIds: [],
          firstSeen: ep.created_at || ep.timestamp,
          lastSeen: ep.created_at || ep.timestamp,
        });
      }
      const node = entityMap.get(key);
      node.count += 1;
      node.episodeIds.push(ep.id || ep.episode_id);
      if (type && node.type === 'DEFAULT') node.type = type;
      const ts = ep.created_at || ep.timestamp;
      if (ts && ts < node.firstSeen) node.firstSeen = ts;
      if (ts && ts > node.lastSeen) node.lastSeen = ts;
      names.push(key);
    }

    // Build co-occurrence edges
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const edgeKey = [names[i], names[j]].sort().join('|');
        if (!edgeMap.has(edgeKey)) {
          edgeMap.set(edgeKey, { source: names[i], target: names[j], weight: 0 });
        }
        edgeMap.get(edgeKey).weight += 1;
      }
    }
  }

  const nodes = Array.from(entityMap.values());
  const links = Array.from(edgeMap.values());

  return { nodes, links };
}

/** Compute node radius from mention count */
function nodeRadius(count) {
  return Math.max(14, Math.min(36, 10 + Math.sqrt(count) * 6));
}

export default function GraphView({ episodes, loading, refreshKey }) {
  const svgRef = useRef(null);
  const simRef = useRef(null);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState(null);
  const [editNode, setEditNode] = useState(null); // node being edited in EntityEditSheet
  const [mutating, setMutating] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  // Build graph from episodes
  useEffect(() => {
    if (!episodes || episodes.length === 0) {
      setGraphData({ nodes: [], links: [] });
      return;
    }
    const data = buildGraphFromEpisodes(episodes);
    setGraphData(data);
    setSelectedNode(null);
    setEditNode(null);
  }, [episodes, refreshKey]);

  // ── Entity mutation operations ────────────────────────────

  /** Rename entity: update node id/name and re-key edges */
  const handleRename = useCallback(async (entityId, newName) => {
    const normalizedNew = newName.trim();
    const newKey = normalizedNew.toLowerCase();

    // Check for collision
    const collision = graphData.nodes.find((n) => n.id === newKey && n.id !== entityId);
    if (collision) return false;

    setMutating(true);
    setGraphData((prev) => {
      const newNodes = prev.nodes.map((n) =>
        n.id === entityId ? { ...n, id: newKey, name: normalizedNew } : n
      );
      const newLinks = prev.links.map((l) => {
        let source = l.source === entityId ? newKey : l.source;
        let target = l.target === entityId ? newKey : l.target;
        return { ...l, source, target };
      });
      return { nodes: newNodes, links: newLinks };
    });
    setMutating(false);
    return true;
  }, [graphData.nodes]);

  /** Delete entity and all its edges */
  const handleDelete = useCallback(async (entityId) => {
    setMutating(true);
    setGraphData((prev) => ({
      nodes: prev.nodes.filter((n) => n.id !== entityId),
      links: prev.links.filter(
        (l) => l.source !== entityId && l.target !== entityId
      ),
    }));
    setSelectedNode(null);
    setMutating(false);
    return true;
  }, []);

  /** Merge: absorb removeId into keepId */
  const handleMerge = useCallback(async (keepId, removeId) => {
    setMutating(true);
    setGraphData((prev) => {
      const keepNode = prev.nodes.find((n) => n.id === keepId);
      const removeNode = prev.nodes.find((n) => n.id === removeId);
      if (!keepNode || !removeNode) return prev;

      // Merge node data
      const mergedNode = {
        ...keepNode,
        count: keepNode.count + removeNode.count,
        episodeIds: [...(keepNode.episodeIds || []), ...(removeNode.episodeIds || [])],
        firstSeen: keepNode.firstSeen && removeNode.firstSeen
          ? (keepNode.firstSeen < removeNode.firstSeen ? keepNode.firstSeen : removeNode.firstSeen)
          : keepNode.firstSeen || removeNode.firstSeen,
        lastSeen: keepNode.lastSeen && removeNode.lastSeen
          ? (keepNode.lastSeen > removeNode.lastSeen ? keepNode.lastSeen : removeNode.lastSeen)
          : keepNode.lastSeen || removeNode.lastSeen,
      };

      const newNodes = prev.nodes
        .filter((n) => n.id !== removeId)
        .map((n) => (n.id === keepId ? mergedNode : n));

      // Transfer and merge edges
      const edgeAcc = new Map();
      for (const l of prev.links) {
        let source = l.source === removeId ? keepId : l.source;
        let target = l.target === removeId ? keepId : l.target;
        if (source === target) continue; // skip self-loops

        const key = [source, target].sort().join('|');
        if (edgeAcc.has(key)) {
          edgeAcc.get(key).weight += l.weight;
        } else {
          edgeAcc.set(key, {
            source: source < target ? source : target,
            target: source < target ? target : source,
            weight: l.weight,
          });
        }
      }

      return { nodes: newNodes, links: Array.from(edgeAcc.values()) };
    });
    setSelectedNode((prev) => (prev?.id === removeId ? null : prev));
    setMutating(false);
    return true;
  }, []);

  /** Open edit sheet for a node */
  const handleOpenEdit = useCallback((node) => {
    // Map node to EntityEditSheet format
    setEditNode({
      id: node.id,
      name: node.name,
      type: node.type?.toLowerCase() || 'unknown',
      mentions: node.count || 0,
      firstSeen: node.firstSeen,
      lastSeen: node.lastSeen,
    });
  }, []);

  const handleCloseEdit = useCallback(() => {
    setEditNode(null);
  }, []);

  // Measure container
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const parent = svg.parentElement;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setDimensions({ width, height });
        }
      }
    });
    ro.observe(parent);
    return () => ro.disconnect();
  }, []);

  // Run d3-force simulation
  useEffect(() => {
    const { nodes, links } = graphData;
    if (nodes.length === 0 || dimensions.width === 0) return;

    // Clone nodes to avoid mutating state
    const simNodes = nodes.map((n) => ({ ...n }));
    const simLinks = links.map((l) => ({
      ...l,
      source: l.source,
      target: l.target,
    }));

    const sim = forceSimulation(simNodes)
      .force(
        'link',
        forceLink(simLinks)
          .id((d) => d.id)
          .distance(80)
          .strength((d) => Math.min(1, 0.3 + d.weight * 0.15))
      )
      .force('charge', forceManyBody().strength(-200).distanceMax(300))
      .force('center', forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force('collide', forceCollide().radius((d) => nodeRadius(d.count) + 6))
      .force('x', forceX(dimensions.width / 2).strength(0.05))
      .force('y', forceY(dimensions.height / 2).strength(0.05))
      .alphaDecay(0.02)
      .on('tick', () => {
        // Clamp nodes within bounds
        for (const n of simNodes) {
          const r = nodeRadius(n.count);
          n.x = Math.max(r, Math.min(dimensions.width - r, n.x));
          n.y = Math.max(r, Math.min(dimensions.height - r, n.y));
        }
        // Update DOM directly for performance
        renderGraph(simNodes, simLinks);
      });

    simRef.current = { sim, nodes: simNodes, links: simLinks };

    return () => {
      sim.stop();
      simRef.current = null;
    };
  }, [graphData, dimensions]);

  // Re-render highlights when selection changes
  useEffect(() => {
    if (!simRef.current) return;
    const { nodes, links } = simRef.current;
    renderGraph(nodes, links);
  }, [selectedNode]);

  /** Direct DOM rendering for smooth animation */
  const renderGraph = useCallback(
    (nodes, links) => {
      const svg = svgRef.current;
      if (!svg) return;

      // Connected set for highlight
      const connectedIds = new Set();
      if (selectedNode) {
        connectedIds.add(selectedNode.id);
        for (const l of links) {
          const srcId = typeof l.source === 'object' ? l.source.id : l.source;
          const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
          if (srcId === selectedNode.id) connectedIds.add(tgtId);
          if (tgtId === selectedNode.id) connectedIds.add(srcId);
        }
      }

      // Update links
      const linkEls = svg.querySelectorAll('.graph-link');
      links.forEach((l, i) => {
        const el = linkEls[i];
        if (!el) return;
        const sx = typeof l.source === 'object' ? l.source.x : 0;
        const sy = typeof l.source === 'object' ? l.source.y : 0;
        const tx = typeof l.target === 'object' ? l.target.x : 0;
        const ty = typeof l.target === 'object' ? l.target.y : 0;
        el.setAttribute('x1', sx);
        el.setAttribute('y1', sy);
        el.setAttribute('x2', tx);
        el.setAttribute('y2', ty);

        if (selectedNode) {
          const srcId = typeof l.source === 'object' ? l.source.id : l.source;
          const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
          const isConnected =
            srcId === selectedNode.id || tgtId === selectedNode.id;
          el.setAttribute('opacity', isConnected ? '0.8' : '0.08');
          el.setAttribute('stroke-width', isConnected ? Math.max(1.5, l.weight) : '0.5');
        } else {
          el.setAttribute('opacity', '0.25');
          el.setAttribute('stroke-width', Math.max(1, l.weight * 0.8));
        }
      });

      // Update nodes
      const nodeEls = svg.querySelectorAll('.graph-node-group');
      nodes.forEach((n, i) => {
        const el = nodeEls[i];
        if (!el) return;
        el.setAttribute('transform', `translate(${n.x},${n.y})`);

        if (selectedNode) {
          const isSelected = n.id === selectedNode.id;
          const isConnected = connectedIds.has(n.id);
          el.setAttribute('opacity', isConnected ? '1' : '0.15');
          const circle = el.querySelector('circle');
          if (circle && isSelected) {
            circle.setAttribute('stroke', '#fff');
            circle.setAttribute('stroke-width', '3');
          } else if (circle) {
            circle.setAttribute('stroke', 'none');
            circle.setAttribute('stroke-width', '0');
          }
        } else {
          el.setAttribute('opacity', '1');
          const circle = el.querySelector('circle');
          if (circle) {
            circle.setAttribute('stroke', 'none');
            circle.setAttribute('stroke-width', '0');
          }
        }
      });
    },
    [selectedNode]
  );

  /** Handle node tap */
  const handleNodeTap = useCallback(
    (node) => {
      if (selectedNode && selectedNode.id === node.id) {
        setSelectedNode(null);
      } else {
        setSelectedNode(node);
      }
    },
    [selectedNode]
  );

  /** Get connected entities for detail panel */
  const connectedEntities = useMemo(() => {
    if (!selectedNode || !graphData.links.length) return [];
    return graphData.links
      .filter((l) => l.source === selectedNode.id || l.target === selectedNode.id)
      .map((l) => {
        const otherId = l.source === selectedNode.id ? l.target : l.source;
        const otherNode = graphData.nodes.find((n) => n.id === otherId);
        return { ...otherNode, edgeWeight: l.weight };
      })
      .filter(Boolean)
      .sort((a, b) => b.edgeWeight - a.edgeWeight);
  }, [selectedNode, graphData]);

  // Show empty state
  if (!loading && graphData.nodes.length === 0) {
    return (
      <div className="memory-subview memory-subview--graph" id="panel-graph" role="tabpanel">
        <div className="memory-subview__empty empty-state">
          <Share2 size={40} className="empty-state__icon" aria-hidden="true" />
          <h3 className="empty-state__title">Knowledge Graph</h3>
          <p className="empty-state__description">
            An interactive visualization of your entities and relationships.
            Start capturing blurts to see your knowledge graph grow.
          </p>
        </div>
      </div>
    );
  }

  if (loading && graphData.nodes.length === 0) {
    return (
      <div className="memory-subview memory-subview--graph" id="panel-graph" role="tabpanel">
        <div className="memory-subview__empty empty-state">
          <Loader size={28} className="empty-state__icon graph-spinner" aria-hidden="true" />
          <p className="empty-state__description">Loading knowledge graph...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="memory-subview memory-subview--graph" id="panel-graph" role="tabpanel">
      <div className="graph-container">
        {/* Stats bar */}
        <div className="graph-stats">
          <span className="graph-stats__item">
            <Hash size={12} aria-hidden="true" />
            {graphData.nodes.length} entities
          </span>
          <span className="graph-stats__item">
            <ArrowRight size={12} aria-hidden="true" />
            {graphData.links.length} connections
          </span>
        </div>

        {/* SVG canvas */}
        <svg
          ref={svgRef}
          className="graph-svg"
          width={dimensions.width || '100%'}
          height={dimensions.height || '100%'}
          viewBox={
            dimensions.width
              ? `0 0 ${dimensions.width} ${dimensions.height}`
              : undefined
          }
        >
          {/* Links */}
          <g className="graph-links">
            {graphData.links.map((l, i) => (
              <line
                key={`${l.source}-${l.target}-${i}`}
                className="graph-link"
                stroke="var(--graph-edge, rgba(99, 144, 255, 0.25))"
                strokeWidth={Math.max(1, l.weight * 0.8)}
                strokeLinecap="round"
              />
            ))}
          </g>

          {/* Nodes */}
          <g className="graph-nodes">
            {graphData.nodes.map((node) => {
              const r = nodeRadius(node.count);
              const color = getTypeColor(node.type);
              // Truncate long labels
              const label =
                node.name.length > 12
                  ? node.name.slice(0, 11) + '\u2026'
                  : node.name;

              return (
                <g
                  key={node.id}
                  className="graph-node-group"
                  role="button"
                  tabIndex={0}
                  aria-label={`Entity: ${node.name}, ${node.count} mentions`}
                  onClick={() => handleNodeTap(node)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleNodeTap(node);
                    }
                  }}
                  style={{ cursor: 'pointer' }}
                >
                  {/* Touch target (invisible, 44px min) */}
                  <circle
                    r={Math.max(22, r + 4)}
                    fill="transparent"
                    className="graph-node__touch-target"
                  />
                  {/* Glow ring for selected */}
                  <circle
                    r={r + 4}
                    fill="none"
                    stroke={color}
                    strokeWidth="0"
                    opacity="0.3"
                    className="graph-node__glow"
                  />
                  {/* Main circle */}
                  <circle
                    r={r}
                    fill={color}
                    opacity="0.85"
                    className="graph-node__circle"
                  />
                  {/* Label */}
                  <text
                    y={r + 14}
                    textAnchor="middle"
                    className="graph-node__label"
                    fill="var(--text-primary, #e8eaed)"
                    fontSize="11"
                    fontWeight="500"
                    pointerEvents="none"
                  >
                    {label}
                  </text>
                  {/* Count badge */}
                  <text
                    textAnchor="middle"
                    dominantBaseline="central"
                    className="graph-node__count"
                    fill="#fff"
                    fontSize={r > 20 ? '12' : '10'}
                    fontWeight="700"
                    pointerEvents="none"
                  >
                    {node.count}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>

        {/* Tap hint */}
        {!selectedNode && graphData.nodes.length > 0 && (
          <div className="graph-hint" aria-hidden="true">
            Tap a node to explore
          </div>
        )}
      </div>

      {/* Entity detail panel */}
      {selectedNode && (
        <EntityDetailPanel
          node={selectedNode}
          connectedEntities={connectedEntities}
          onClose={() => setSelectedNode(null)}
          onSelectEntity={(entity) => setSelectedNode(entity)}
          onEdit={() => handleOpenEdit(selectedNode)}
        />
      )}

      {/* Entity edit sheet (merge / rename / delete) */}
      <EntityEditSheet
        entity={editNode}
        allNodes={graphData.nodes.map((n) => ({
          id: n.id,
          name: n.name,
          type: n.type?.toLowerCase() || 'unknown',
          mentions: n.count || 0,
          firstSeen: n.firstSeen,
          lastSeen: n.lastSeen,
        }))}
        onRename={handleRename}
        onMerge={handleMerge}
        onDelete={handleDelete}
        onClose={handleCloseEdit}
        mutating={mutating}
      />
    </div>
  );
}

/** Entity detail panel / popover */
function EntityDetailPanel({ node, connectedEntities, onClose, onSelectEntity, onEdit }) {
  const color = getTypeColor(node.type);

  const formatDate = (ts) => {
    if (!ts) return 'Unknown';
    try {
      return new Date(ts).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
      });
    } catch {
      return ts;
    }
  };

  return (
    <div className="entity-panel" role="dialog" aria-label={`Entity details: ${node.name}`}>
      <div className="entity-panel__header">
        <div className="entity-panel__title-row">
          <div
            className="entity-panel__dot"
            style={{ background: color }}
            aria-hidden="true"
          />
          <h3 className="entity-panel__name">{node.name}</h3>
          <button
            className="entity-panel__edit-btn"
            onClick={onEdit}
            aria-label="Edit entity"
          >
            <Pencil size={16} />
          </button>
          <button
            className="entity-panel__close"
            onClick={onClose}
            aria-label="Close entity panel"
          >
            <X size={18} />
          </button>
        </div>
        <div className="entity-panel__meta">
          <span
            className="entity-panel__type-badge"
            style={{
              color,
              background: `color-mix(in srgb, ${color} 14%, transparent)`,
            }}
          >
            {node.type || 'Entity'}
          </span>
          <span className="entity-panel__stat">
            <Hash size={12} aria-hidden="true" />
            {node.count} mention{node.count !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      <div className="entity-panel__body">
        {/* Timeline info */}
        <div className="entity-panel__section">
          <div className="entity-panel__section-label">Activity</div>
          <div className="entity-panel__dates">
            <div className="entity-panel__date-row">
              <Clock size={13} aria-hidden="true" />
              <span>First seen: {formatDate(node.firstSeen)}</span>
            </div>
            <div className="entity-panel__date-row">
              <Clock size={13} aria-hidden="true" />
              <span>Last seen: {formatDate(node.lastSeen)}</span>
            </div>
          </div>
        </div>

        {/* Connected entities */}
        {connectedEntities.length > 0 && (
          <div className="entity-panel__section">
            <div className="entity-panel__section-label">
              Connected ({connectedEntities.length})
            </div>
            <div className="entity-panel__connections">
              {connectedEntities.map((ent) => (
                <button
                  key={ent.id}
                  className="entity-panel__connection-chip"
                  onClick={() => onSelectEntity(ent)}
                  style={{
                    borderColor: `color-mix(in srgb, ${getTypeColor(ent.type)} 30%, transparent)`,
                  }}
                >
                  <span
                    className="entity-panel__connection-dot"
                    style={{ background: getTypeColor(ent.type) }}
                    aria-hidden="true"
                  />
                  <span className="entity-panel__connection-name">{ent.name}</span>
                  <span className="entity-panel__connection-weight">
                    {ent.edgeWeight}x
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Episode IDs preview */}
        {node.episodeIds && node.episodeIds.length > 0 && (
          <div className="entity-panel__section">
            <div className="entity-panel__section-label">
              Episodes ({node.episodeIds.length})
            </div>
            <div className="entity-panel__episode-ids">
              {node.episodeIds.slice(0, 5).map((id, i) => (
                <span key={i} className="entity-panel__episode-id">
                  {id ? id.slice(0, 8) + '\u2026' : 'N/A'}
                </span>
              ))}
              {node.episodeIds.length > 5 && (
                <span className="entity-panel__more">
                  +{node.episodeIds.length - 5} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
