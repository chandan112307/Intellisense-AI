"""
Knowledge Graph Layer.

Builds a concept graph from documents, maps relationships and prerequisites,
and supports graph-based retrieval and reasoning.
"""

import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import log_error, log_info, log_warning


@dataclass
class ConceptNode:
    """Represents a single concept in the knowledge graph."""

    concept_id: str
    name: str
    description: str
    document_ids: list[str]
    importance: float  # 0-1


@dataclass
class ConceptEdge:
    """Represents a directional relationship between two concepts."""

    source_id: str
    target_id: str
    relationship: str  # "prerequisite", "related", "part_of", "extends"
    weight: float  # 0-1


_VALID_RELATIONSHIPS = {"prerequisite", "related", "part_of", "extends"}

# Patterns used by extract_concepts_from_text
_DEFINITION_PATTERN = re.compile(
    r"([A-Z][a-zA-Z\s]{1,60}?)(?:\s+(?:is a|refers to|is defined as|means)\s+)",
)
_CAPITALISED_PHRASE_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
)


class KnowledgeGraph:
    """In-memory concept graph with BFS traversal and query enrichment."""

    def __init__(self) -> None:
        self.nodes: dict[str, ConceptNode] = {}
        self.edges: list[ConceptEdge] = []
        self.adjacency: dict[str, list[str]] = {}
        self._name_index: dict[str, str] = {}  # lower-cased name -> concept_id
        log_info("KnowledgeGraph initialised")

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_concept(
        self,
        name: str,
        description: str,
        document_ids: list[str] | None = None,
        importance: float = 0.5,
    ) -> str:
        """Add a concept node and return its *concept_id*.

        If a concept with the same name (case-insensitive) already exists the
        existing id is returned and document_ids are merged.
        """
        importance = max(0.0, min(1.0, importance))
        key = name.strip().lower()

        if key in self._name_index:
            existing = self.nodes[self._name_index[key]]
            for did in document_ids or []:
                if did not in existing.document_ids:
                    existing.document_ids.append(did)
            log_info(f"Concept '{name}' already exists – merged document_ids")
            return existing.concept_id

        concept_id = str(uuid.uuid4())
        node = ConceptNode(
            concept_id=concept_id,
            name=name.strip(),
            description=description,
            document_ids=list(document_ids or []),
            importance=importance,
        )
        self.nodes[concept_id] = node
        self.adjacency[concept_id] = []
        self._name_index[key] = concept_id
        log_info(f"Added concept '{name}' ({concept_id})")
        return concept_id

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship: str,
        weight: float = 0.5,
    ) -> None:
        """Create a directed edge between two concepts identified by name."""
        if relationship not in _VALID_RELATIONSHIPS:
            log_error(
                f"Invalid relationship '{relationship}'. "
                f"Must be one of {_VALID_RELATIONSHIPS}"
            )
            return

        weight = max(0.0, min(1.0, weight))

        source = self.get_concept_by_name(source_name)
        target = self.get_concept_by_name(target_name)

        if source is None:
            log_warning(f"Source concept '{source_name}' not found – skipping edge")
            return
        if target is None:
            log_warning(f"Target concept '{target_name}' not found – skipping edge")
            return

        edge = ConceptEdge(
            source_id=source.concept_id,
            target_id=target.concept_id,
            relationship=relationship,
            weight=weight,
        )
        self.edges.append(edge)
        self.adjacency.setdefault(source.concept_id, []).append(target.concept_id)

        # For "related" edges add the reverse direction as well
        if relationship == "related":
            self.adjacency.setdefault(target.concept_id, []).append(
                source.concept_id
            )

        log_info(
            f"Added edge '{source_name}' --[{relationship}]--> '{target_name}'"
        )

    # ------------------------------------------------------------------
    # Concept extraction
    # ------------------------------------------------------------------

    def extract_concepts_from_text(
        self, text: str, document_id: str
    ) -> list[str]:
        """Extract key concepts from *text* using regex heuristics.

        Returns a list of concept names that were added (or merged) into the
        graph.
        """
        concepts_found: dict[str, str] = {}  # lower-name -> original name

        # 1. Definitions: "X is a …", "X refers to …"
        for match in _DEFINITION_PATTERN.finditer(text):
            name = match.group(1).strip()
            if len(name) > 2:
                concepts_found[name.lower()] = name

        # 2. Capitalised multi-word phrases (likely proper nouns / terms)
        for match in _CAPITALISED_PHRASE_PATTERN.finditer(text):
            name = match.group(1).strip()
            if len(name) > 2:
                concepts_found[name.lower()] = name

        added: list[str] = []
        for name in concepts_found.values():
            self.add_concept(
                name=name,
                description=f"Extracted from document {document_id}",
                document_ids=[document_id],
            )
            added.append(name)

        log_info(
            f"Extracted {len(added)} concept(s) from document '{document_id}'"
        )
        return added

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_concept_by_name(self, name: str) -> Optional[ConceptNode]:
        """Look up a concept by its name (case-insensitive)."""
        cid = self._name_index.get(name.strip().lower())
        if cid is None:
            return None
        return self.nodes.get(cid)

    def get_prerequisites(self, concept_name: str) -> list[ConceptNode]:
        """Return all direct prerequisite concepts of *concept_name*."""
        target = self.get_concept_by_name(concept_name)
        if target is None:
            log_warning(f"Concept '{concept_name}' not found for prerequisites")
            return []

        prereqs: list[ConceptNode] = []
        for edge in self.edges:
            if (
                edge.target_id == target.concept_id
                and edge.relationship == "prerequisite"
            ):
                node = self.nodes.get(edge.source_id)
                if node is not None:
                    prereqs.append(node)
        return prereqs

    def get_related_concepts(
        self, concept_name: str, max_depth: int = 2
    ) -> list[ConceptNode]:
        """BFS traversal returning concepts up to *max_depth* hops away."""
        start = self.get_concept_by_name(concept_name)
        if start is None:
            log_warning(f"Concept '{concept_name}' not found for BFS traversal")
            return []

        visited: set[str] = {start.concept_id}
        queue: deque[tuple[str, int]] = deque([(start.concept_id, 0)])
        result: list[ConceptNode] = []

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for neighbour_id in self.adjacency.get(current_id, []):
                if neighbour_id not in visited:
                    visited.add(neighbour_id)
                    node = self.nodes.get(neighbour_id)
                    if node is not None:
                        result.append(node)
                        queue.append((neighbour_id, depth + 1))

        return result

    def get_learning_path(
        self, from_concept: str, to_concept: str
    ) -> list[ConceptNode]:
        """Find the shortest path between two concepts (BFS)."""
        start = self.get_concept_by_name(from_concept)
        end = self.get_concept_by_name(to_concept)

        if start is None or end is None:
            log_warning(
                f"Cannot compute learning path: "
                f"'{from_concept}' or '{to_concept}' not found"
            )
            return []

        if start.concept_id == end.concept_id:
            return [start]

        visited: set[str] = {start.concept_id}
        queue: deque[list[str]] = deque([[start.concept_id]])

        while queue:
            path = queue.popleft()
            current_id = path[-1]

            for neighbour_id in self.adjacency.get(current_id, []):
                if neighbour_id in visited:
                    continue
                new_path = path + [neighbour_id]

                if neighbour_id == end.concept_id:
                    return [
                        self.nodes[cid]
                        for cid in new_path
                        if cid in self.nodes
                    ]

                visited.add(neighbour_id)
                queue.append(new_path)

        log_info(
            f"No path found from '{from_concept}' to '{to_concept}'"
        )
        return []

    # ------------------------------------------------------------------
    # Query enrichment
    # ------------------------------------------------------------------

    def enrich_query_with_graph(self, query: str) -> str:
        """Augment *query* with names of related concepts for better retrieval."""
        extra_terms: list[str] = []

        for name, cid in self._name_index.items():
            if name in query.lower():
                node = self.nodes[cid]
                related = self.get_related_concepts(node.name, max_depth=1)
                extra_terms.extend(r.name for r in related)

        if not extra_terms:
            return query

        unique_terms = list(dict.fromkeys(extra_terms))
        enriched = f"{query} [Related: {', '.join(unique_terms)}]"
        log_info(f"Enriched query with {len(unique_terms)} related concept(s)")
        return enriched

    # ------------------------------------------------------------------
    # Statistics & serialisation
    # ------------------------------------------------------------------

    def get_graph_stats(self) -> dict:
        """Return basic statistics about the graph."""
        connection_counts: dict[str, int] = {}
        for cid, neighbours in self.adjacency.items():
            node = self.nodes.get(cid)
            if node is not None:
                connection_counts[node.name] = len(neighbours)

        sorted_concepts = sorted(
            connection_counts.items(), key=lambda x: x[1], reverse=True
        )
        return {
            "nodes_count": len(self.nodes),
            "edges_count": len(self.edges),
            "most_connected_concepts": sorted_concepts[:10],
        }

    def to_dict(self) -> dict:
        """Serialise the entire graph to a plain dictionary."""
        return {
            "nodes": {
                cid: {
                    "concept_id": n.concept_id,
                    "name": n.name,
                    "description": n.description,
                    "document_ids": n.document_ids,
                    "importance": n.importance,
                }
                for cid, n in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "relationship": e.relationship,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from a dictionary produced by *to_dict*."""
        graph = cls()

        for cid, ndata in data.get("nodes", {}).items():
            node = ConceptNode(
                concept_id=ndata["concept_id"],
                name=ndata["name"],
                description=ndata["description"],
                document_ids=ndata["document_ids"],
                importance=ndata["importance"],
            )
            graph.nodes[cid] = node
            graph.adjacency[cid] = []
            graph._name_index[node.name.strip().lower()] = cid

        for edata in data.get("edges", []):
            edge = ConceptEdge(
                source_id=edata["source_id"],
                target_id=edata["target_id"],
                relationship=edata["relationship"],
                weight=edata["weight"],
            )
            graph.edges.append(edge)
            graph.adjacency.setdefault(edge.source_id, []).append(edge.target_id)
            if edge.relationship == "related":
                graph.adjacency.setdefault(edge.target_id, []).append(
                    edge.source_id
                )

        log_info(
            f"Loaded graph from dict – "
            f"{len(graph.nodes)} node(s), {len(graph.edges)} edge(s)"
        )
        return graph
