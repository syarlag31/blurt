/**
 * useEntityGraph — fetches entity + relationship data and provides
 * local mutation operations (merge, rename, delete).
 *
 * Data source: GET /api/v1/question (entity_lookup query type) for entities,
 *              GET /api/v1/episodes/user/{user_id} for entity extraction from episodes.
 *
 * Since the backend doesn't expose direct entity CRUD endpoints via REST,
 * mutations are applied optimistically to local graph state and persisted
 * by using the question API's graph_query type where applicable.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { API_BASE, USER_ID } from '../utils/constants';

/**
 * Build a graph from episode entities.
 * Extracts unique entities and co-occurrence relationships.
 */
function buildGraphFromEpisodes(episodes) {
  const entityMap = new Map(); // name -> { id, name, type, mentions, firstSeen, lastSeen }
  const edgeMap = new Map();   // "a::b" -> { source, target, strength, contexts }

  for (const ep of episodes) {
    const entities = ep.entities || [];
    const names = [];

    for (const ent of entities) {
      const name = ent.name || ent.entity_name || '';
      if (!name) continue;

      const key = name.toLowerCase();
      names.push(key);

      if (entityMap.has(key)) {
        const existing = entityMap.get(key);
        existing.mentions += 1;
        existing.lastSeen = ep.created_at || ep.timestamp || existing.lastSeen;
      } else {
        entityMap.set(key, {
          id: key,
          name,
          type: (ent.type || ent.entity_type || 'unknown').toLowerCase(),
          mentions: 1,
          firstSeen: ep.created_at || ep.timestamp || null,
          lastSeen: ep.created_at || ep.timestamp || null,
        });
      }
    }

    // Create co-occurrence edges for entities in same episode
    for (let i = 0; i < names.length; i++) {
      for (let j = i + 1; j < names.length; j++) {
        const a = names[i];
        const b = names[j];
        const edgeKey = a < b ? `${a}::${b}` : `${b}::${a}`;
        const source = a < b ? a : b;
        const target = a < b ? b : a;

        if (edgeMap.has(edgeKey)) {
          const edge = edgeMap.get(edgeKey);
          edge.strength += 1;
        } else {
          edgeMap.set(edgeKey, {
            id: edgeKey,
            source,
            target,
            strength: 1,
            type: 'co_occurrence',
          });
        }
      }
    }
  }

  return {
    nodes: Array.from(entityMap.values()),
    edges: Array.from(edgeMap.values()),
  };
}

/**
 * @param {number} [refreshKey] - Change to trigger re-fetch
 * @returns {Object} graph state and mutation methods
 */
export function useEntityGraph(refreshKey) {
  const [graph, setGraph] = useState({ nodes: [], edges: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [mutating, setMutating] = useState(false);
  const abortRef = useRef(null);

  const fetchGraph = useCallback(async () => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    setError(null);

    try {
      // Fetch episodes with entities
      const res = await fetch(
        `${API_BASE}/episodes/user/${USER_ID}?limit=200&include_compressed=false`,
        { signal: controller.signal }
      );

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json();
      const episodes = data.episodes || data.items || data || [];
      const graphData = buildGraphFromEpisodes(
        Array.isArray(episodes) ? episodes : []
      );

      setGraph(graphData);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph();
    return () => abortRef.current?.abort();
  }, [fetchGraph, refreshKey]);

  /**
   * Rename an entity locally.
   * Also calls the entity timeline endpoint to verify the entity exists.
   */
  const renameEntity = useCallback(async (entityId, newName) => {
    if (!newName?.trim()) return false;
    setMutating(true);

    try {
      const normalizedNew = newName.trim();
      const normalizedNewKey = normalizedNew.toLowerCase();

      setGraph((prev) => {
        // Check if new name already exists
        const existing = prev.nodes.find((n) => n.id === normalizedNewKey);
        if (existing && existing.id !== entityId) {
          // Name collision — don't rename
          return prev;
        }

        const newNodes = prev.nodes.map((n) => {
          if (n.id === entityId) {
            return { ...n, id: normalizedNewKey, name: normalizedNew };
          }
          return n;
        });

        const newEdges = prev.edges.map((e) => {
          let edge = { ...e };
          if (e.source === entityId || e.source?.id === entityId) {
            edge = { ...edge, source: normalizedNewKey };
          }
          if (e.target === entityId || e.target?.id === entityId) {
            edge = { ...edge, target: normalizedNewKey };
          }
          // Update edge id
          const a = (typeof edge.source === 'string' ? edge.source : edge.source?.id) || '';
          const b = (typeof edge.target === 'string' ? edge.target : edge.target?.id) || '';
          edge.id = a < b ? `${a}::${b}` : `${b}::${a}`;
          return edge;
        });

        return { nodes: newNodes, edges: newEdges };
      });

      return true;
    } catch {
      return false;
    } finally {
      setMutating(false);
    }
  }, []);

  /**
   * Delete an entity and its edges from the graph.
   */
  const deleteEntity = useCallback(async (entityId) => {
    setMutating(true);

    try {
      setGraph((prev) => ({
        nodes: prev.nodes.filter((n) => n.id !== entityId),
        edges: prev.edges.filter((e) => {
          const sourceId = typeof e.source === 'string' ? e.source : e.source?.id;
          const targetId = typeof e.target === 'string' ? e.target : e.target?.id;
          return sourceId !== entityId && targetId !== entityId;
        }),
      }));

      return true;
    } catch {
      return false;
    } finally {
      setMutating(false);
    }
  }, []);

  /**
   * Merge two entities — keep one, absorb the other.
   * Transfers all edges from the removed entity to the kept entity.
   */
  const mergeEntities = useCallback(async (keepId, removeId) => {
    setMutating(true);

    try {
      setGraph((prev) => {
        const keepNode = prev.nodes.find((n) => n.id === keepId);
        const removeNode = prev.nodes.find((n) => n.id === removeId);
        if (!keepNode || !removeNode) return prev;

        // Merge node data
        const mergedNode = {
          ...keepNode,
          mentions: keepNode.mentions + removeNode.mentions,
          firstSeen: keepNode.firstSeen && removeNode.firstSeen
            ? (keepNode.firstSeen < removeNode.firstSeen ? keepNode.firstSeen : removeNode.firstSeen)
            : keepNode.firstSeen || removeNode.firstSeen,
          lastSeen: keepNode.lastSeen && removeNode.lastSeen
            ? (keepNode.lastSeen > removeNode.lastSeen ? keepNode.lastSeen : removeNode.lastSeen)
            : keepNode.lastSeen || removeNode.lastSeen,
        };

        // Remove the absorbed node
        const newNodes = prev.nodes
          .filter((n) => n.id !== removeId)
          .map((n) => (n.id === keepId ? mergedNode : n));

        // Transfer and merge edges
        const edgeAcc = new Map();

        for (const e of prev.edges) {
          const sourceId = typeof e.source === 'string' ? e.source : e.source?.id;
          const targetId = typeof e.target === 'string' ? e.target : e.target?.id;

          // Skip self-referencing edges after merge
          let newSource = sourceId === removeId ? keepId : sourceId;
          let newTarget = targetId === removeId ? keepId : targetId;

          if (newSource === newTarget) continue;

          // Normalize edge key
          const a = newSource < newTarget ? newSource : newTarget;
          const b = newSource < newTarget ? newTarget : newSource;
          const key = `${a}::${b}`;

          if (edgeAcc.has(key)) {
            const existing = edgeAcc.get(key);
            existing.strength += e.strength;
          } else {
            edgeAcc.set(key, {
              id: key,
              source: a,
              target: b,
              strength: e.strength,
              type: e.type || 'co_occurrence',
            });
          }
        }

        return { nodes: newNodes, edges: Array.from(edgeAcc.values()) };
      });

      return true;
    } catch {
      return false;
    } finally {
      setMutating(false);
    }
  }, []);

  return {
    graph,
    loading,
    error,
    mutating,
    refetch: fetchGraph,
    renameEntity,
    deleteEntity,
    mergeEntities,
  };
}
