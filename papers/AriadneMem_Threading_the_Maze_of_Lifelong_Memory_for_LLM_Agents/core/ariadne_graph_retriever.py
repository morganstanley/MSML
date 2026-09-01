"""
AriadneGraphRetriever - Phase II: Real-Time Structural Reasoning

Paper Reference: Section 2.3 - Phase II: Real-Time Structural Reasoning
Key Features:
1. Fast Paths: O(1) cache lookup for count/list/relation queries
2. Hybrid Retrieval (Eq. 7): Semantic (dense) + Lexical (sparse) search
3. Base Graph Construction (Eq. 8): Entity/temporal edge inference
4. Algorithmic Bridge Discovery (Eq. 9): Steiner tree approximation
5. Multi-Hop Path Mining (Eq. 10): DFS-based reasoning path discovery
"""
from typing import List, Dict, Any, Set, Optional
import re
from datetime import datetime
from collections import defaultdict
import concurrent.futures

# Models & Utils
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config

class GraphPath:
    """
    Data structure representing a minimal subgraph (Nodes + Edges).
    Passed to Stage 4 (Answer Generator) for topology-aware reasoning.
    """
    def __init__(self, nodes: List[MemoryEntry], edges: List[Dict[str, Any]], reasoning_paths: List[List[MemoryEntry]] = None):
        self.nodes = nodes
        self.edges = edges
        self.reasoning_paths = reasoning_paths or []
        # [NEW] Metadata for answer generation
        self.target_entity: Optional[str] = None  # The entity the question is asking about 

class AriadneGraphRetriever:
    """
    AriadneGraphRetriever - Phase II: Real-Time Structural Reasoning
    
    Paper Reference: Section 2.3
    Constructs query-specific evidence graph G_q via:
    1. Hybrid retrieval to find terminal nodes V_term (Eq. 7)
    2. Base graph construction with entity/temporal edges (Eq. 8)
    3. Steiner tree approximation with bridge discovery (Eq. 9)
    4. DFS path mining for reasoning chains (Eq. 10)
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.enhanced_index = None  # Will be set after memory building

    def retrieve(self, query: str) -> GraphPath:
        """
        Execute structural retrieval - Paper Algorithm 1, lines 9-17
        
        Paper Reference: Section 2.3 - Phase II: Real-Time Structural Reasoning
        Pipeline:
        1. Fast paths (cache/regex lookup)
        2. Hybrid retrieval for terminal nodes V_term
        3. Base graph construction G_0
        4. Bridge discovery (Steiner approximation)
        5. Multi-hop path mining via DFS
        """
        # Step 0a: Fast path - enhanced cache lookup (count, list, relation queries)
        if self.enhanced_index:
            cached_result = self._try_enhanced_cache_lookup(query)
            if cached_result:
                print(f"[AriadneMem] Cache hit - answered from enhanced cache")
                return cached_result
        
        # Step 0b: Fast path - attribute lookup (regex-based O(1))
        quick_result = self._try_attribute_lookup(query)
        if quick_result:
            return quick_result
        
        # Step 1: Extract target entity (for entity-aware filtering)
        target_entity = self._extract_target_entity(query)
        if target_entity:
            target_entity = target_entity.lower()
        
        # Step 2: Hybrid Retrieval - V_term = Top-k_sem(q) ∪ Top-k_lex(q) (Eq. 7)
        candidate_nodes = self._hybrid_recall(query)
        
        if not candidate_nodes:
            return GraphPath([], [])
        
        # Step 3: Entity-aware filtering (reduce noise from irrelevant entities)
        if target_entity:
            candidate_nodes = self._filter_by_entity(candidate_nodes, target_entity, query)

        # Step 4: Build inference graph with bridge discovery (Eq. 8-9)
        graph_path = self._build_inference_graph(candidate_nodes)
        
        # Step 5: Apply node budget (control context length)
        graph_path.nodes = self._rank_and_limit_nodes(
            graph_path.nodes, target_entity, query
        )
        
        # Attach metadata for topology-aware synthesis
        graph_path.target_entity = target_entity
        
        return graph_path
    
    def set_enhanced_index(self, enhanced_index):
        """Set enhanced index from memory builder"""
        self.enhanced_index = enhanced_index
        print(f"[Retriever] Enhanced index loaded with {len(enhanced_index.entities)} entities")
    
    def _try_enhanced_cache_lookup(self, query: str) -> Optional[GraphPath]:
        """
        Fast path: O(1) cache lookup for common query patterns
        
        Paper Reference: Section 2.3 - Fast Paths (Heuristic Short-Circuiting)
        Handles: count queries, list queries, relationship queries
        """
        if not self.enhanced_index:
            return None
        
        query_lower = query.lower()
        
        # Pattern 1: Count queries ("how many times", "how many X")
        if 'how many' in query_lower or 'count' in query_lower or 'times' in query_lower:
            result = self._check_count_cache(query, query_lower)
            if result:
                return result
        
        # Pattern 2: "All X" queries ("what are all the books")
        if 'all ' in query_lower or 'every ' in query_lower or 'list ' in query_lower:
            result = self._check_list_cache(query, query_lower)
            if result:
                return result
        
        # Pattern 3: Relationship queries ("both X and Y", "X and Y both")
        if 'both' in query_lower and ' and ' in query_lower:
            result = self._check_relation_cache(query, query_lower)
            if result:
                return result
        
        return None
    
    def _check_count_cache(self, query: str, query_lower: str) -> Optional[GraphPath]:
        """Check if count can be answered from cache"""
        # Extract entity name
        entity_name = self._extract_target_entity(query)
        if not entity_name or entity_name not in self.enhanced_index.entities:
            return None
        
        entity_agg = self.enhanced_index.entities[entity_name]
        
        # Look for action in query
        for action, count in entity_agg.event_counts.items():
            # Match action keywords in query
            action_words = action.replace('_', ' ').split()
            if any(word in query_lower for word in action_words):
                # Create synthetic memory with the count
                synthetic_fact = f"{entity_name} {action.replace('_', ' ')} {count} times"
                node = MemoryEntry(
                    lossless_restatement=synthetic_fact,
                    persons=[entity_name],
                    keywords=[entity_name, action, str(count)],
                    topic="count"
                )
                return GraphPath([node], [])
        
        return None
    
    def _check_list_cache(self, query: str, query_lower: str) -> Optional[GraphPath]:
        """Check if list query can be answered from cache"""
        entity_name = self._extract_target_entity(query)
        if not entity_name or entity_name not in self.enhanced_index.entities:
            return None
        
        entity_agg = self.enhanced_index.entities[entity_name]
        
        # Check attribute sets
        for attr_type, values in entity_agg.attribute_sets.items():
            if attr_type in query_lower or any(word in query_lower for word in attr_type.split('_')):
                if values:
                    values_str = ', '.join(sorted(values))
                    synthetic_fact = f"{entity_name}'s {attr_type}: {values_str}"
                    node = MemoryEntry(
                        lossless_restatement=synthetic_fact,
                        persons=[entity_name],
                        keywords=[entity_name, attr_type] + list(values)[:5],  # Limit keywords
                        topic=attr_type
                    )
                    return GraphPath([node], [])
        
        return None
    
    def _check_relation_cache(self, query: str, query_lower: str) -> Optional[GraphPath]:
        """Check if relationship query can be answered from relation triples"""
        # Extract two entity names
        entities = re.findall(r'\b([A-Z][a-z]+)\b', query)
        if len(entities) < 2:
            return None
        
        entity1, entity2 = entities[0], entities[1]
        
        # Find relations between these entities
        matching_relations = []
        for relation in self.enhanced_index.relations:
            if (relation.subject == entity1 and relation.object == entity2) or \
               (relation.subject == entity2 and relation.object == entity1):
                matching_relations.append(relation)
        
        if matching_relations:
            # Create nodes from relation triples
            nodes = []
            for rel in matching_relations[:3]:  # Limit to 3 relations
                synthetic_fact = f"{rel.subject} {rel.predicate} {rel.object}"
                node = MemoryEntry(
                    lossless_restatement=synthetic_fact,
                    persons=[rel.subject, rel.object],
                    keywords=[rel.subject, rel.predicate, rel.object],
                    topic="relationship",
                    timestamp=rel.timestamp,
                    location=rel.location
                )
                nodes.append(node)
            return GraphPath(nodes, [])
        
        return None
    
    def _try_attribute_lookup(self, query: str) -> Optional[GraphPath]:
        """
        Fast path: Regex-based attribute lookup
        
        Paper Reference: Section 2.3 - Fast Paths
        Handles simple "X's attribute" queries via direct metadata lookup
        No LLM call required
        """
        query_lower = query.lower()
        
        # Extract person name from query
        name_match = re.search(r"\b([A-Z][a-z]+)\b", query)
        if not name_match:
            return None
        person = name_match.group(1)
        
        # Try different attribute types based on query patterns
        attr_checks = []
        
        if 'status' in query_lower or 'relationship' in query_lower or 'married' in query_lower or 'single' in query_lower:
            attr_checks.append(('relationship_status', f"{person}'s relationship status"))
        
        if 'identity' in query_lower or 'gender' in query_lower:
            attr_checks.append(('identity', f"{person}'s identity"))
        
        if 'from' in query_lower or 'origin' in query_lower or 'move' in query_lower or 'country' in query_lower or 'hometown' in query_lower:
            attr_checks.append(('origin', f"{person}'s origin"))
        
        if 'job' in query_lower or 'work' in query_lower or 'occupation' in query_lower or 'profession' in query_lower:
            attr_checks.append(('occupation', f"{person}'s occupation"))
        
        # Try each attribute type
        for attr_type, description in attr_checks:
            if hasattr(self.vector_store, 'query_attribute'):
                value = self.vector_store.query_attribute(person, attr_type)
                if value:
                    # Create synthetic node with the found attribute
                    synthetic_fact = f"{person}'s {attr_type.replace('_', ' ')} is {value}."
                    node = MemoryEntry(
                        lossless_restatement=synthetic_fact,
                        persons=[person],
                        keywords=[person, attr_type.replace('_', ' '), str(value)],
                        topic=attr_type.replace('_', ' ')
                    )
                    graph_path = GraphPath([node], [])
                    graph_path.target_entity = person.lower()
                    return graph_path
        
        return None
    
    def _rank_and_limit_nodes(self, nodes: List[MemoryEntry], target_entity: Optional[str], 
                              query: str) -> List[MemoryEntry]:
        """
        Apply node budget for context length control
        
        Paper Reference: Section 2.3 - Multi-Hop Path Mining & Node Budget
        Prioritizes paths based on relevance score, enforces budget (8-25 nodes)
        """
        if not nodes:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Extract date from query if present
        query_date = None
        date_match = re.search(r'(\d{1,2})\s(st|nd|rd|th)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*(\d{4})?', query_lower)
        if date_match:
            query_date = date_match.group(0)
        
        def score_node(node: MemoryEntry) -> float:
            score = 0.0
            content = (node.lossless_restatement or "").lower()
            
            # 1. Target entity match (highest priority)
            if target_entity:
                persons = [p.lower() for p in (node.persons or [])]
                entities = [e.lower() for e in (node.entities or [])]
                if target_entity in persons or target_entity in content:
                    score += 100
                elif target_entity in entities:
                    score += 80
            
            # 2. Date match (for time-specific queries)
            if query_date and query_date in content:
                score += 80
            
            # 3. Keyword overlap (with entities/keywords boost)
            node_words = set(content.split())
            overlap = len(query_words & node_words)
            score += overlap * 10
            
            # Boost for keyword field match (high-precision signals)
            keywords = [k.lower() for k in (node.keywords or [])]
            keyword_overlap = len(query_words & set(keywords))
            score += keyword_overlap * 15
            
            # 4. Recency bonus (prefer nodes with timestamps)
            if node.timestamp:
                score += 5
            
            return score
        
        # Score all nodes
        scored_nodes = [(node, score_node(node)) for node in nodes]
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Node budget (Paper Section 2.3): 8-25 nodes for context control
        MIN_NODES = 8    # Minimum for multi-hop reasoning
        MAX_NODES = 25   # Maximum to control token cost
        
        max_score = scored_nodes[0][1] if scored_nodes else 0
        threshold = max_score * 0.10  # 10% threshold - more lenient to get quality nodes within budget
        
        # Filter by threshold
        relevant_nodes = [(n, s) for n, s in scored_nodes if s >= threshold]
        
        # Apply strict bounds
        if len(relevant_nodes) < MIN_NODES:
            relevant_nodes = scored_nodes[:min(MIN_NODES, len(scored_nodes))]
        elif len(relevant_nodes) > MAX_NODES:
            relevant_nodes = relevant_nodes[:MAX_NODES]
        
        return [node for node, _ in relevant_nodes]
    
    def _extract_target_entity(self, query: str) -> Optional[str]:
        """
        Extract the primary entity the question is asking about.
        Uses generic patterns - no hardcoded word lists.
        """
        # Pattern 1: Possessive "X's Y" - extract the whole phrase
        possessive_match = re.search(r"([A-Z][a-z]+)'s\s+(\w+)", query)
        if possessive_match:
            return possessive_match.group(0).lower()
        
        # Pattern 2: Find first capitalized name (likely the subject)
        name_match = re.search(r"\b([A-Z][a-z]{2,})\b", query)
        if name_match:
            return name_match.group(1).lower()
        
        return None
    
    def _extract_person_from_query(self, query: str) -> Optional[str]:
        """
        Extract person name from query for structured search.
        Simple: just find the first capitalized word that looks like a name.
        """
        # Find capitalized words (likely names)
        name_match = re.search(r"\b([A-Z][a-z]{2,})\b", query)
        if name_match:
            return name_match.group(1)
        return None
    
    def _filter_by_entity(
        self, 
        nodes: List[MemoryEntry], 
        target_entity: str,
        query: str
    ) -> List[MemoryEntry]:
        """
        Filter and rank nodes based on relevance to the query.
        [ENHANCED] Better handling of possessive queries like "Person X's attribute Y".
        """
        query_lower = query.lower()
        target_lower = target_entity.lower() if target_entity else ""
        
        # [CRITICAL for Category 1] Detect possessive pattern: "X's Y"
        possessive_match = re.search(r"([A-Za-z]+)'s\s+(\w+)", query)
        person_name = None
        attribute_word = None
        if possessive_match:
            person_name = possessive_match.group(1).lower()
            attribute_word = possessive_match.group(2).lower()
        
        # Extract meaningful words from query (length > 2 to skip articles/prepositions)
        query_words = set(w.strip('?.,!') for w in query_lower.split() if len(w) > 2)
        
        scored_nodes = []
        for node in nodes:
            score = 0
            content = (node.lossless_restatement or "").lower()
            persons = [p.lower() for p in (node.persons or [])]
            entities = [e.lower() for e in (node.entities or [])]
            keywords = [k.lower() for k in (node.keywords or [])]
            
            all_text = content + " " + " ".join(persons) + " " + " ".join(entities) + " " + " ".join(keywords)
            
            # [CRITICAL for Category 1] Boost for possessive match using co-occurrence
            # "Person's attribute" -> prioritize facts where person + attribute appear together
            if person_name and attribute_word:
                has_person = person_name in all_text or person_name in persons
                
                # Check attribute word variations (generic morphological)
                attr_variations = self._get_word_variations(attribute_word)
                has_attribute = any(v in all_text for v in attr_variations)
                
                # [KEY] Check if attribute appears in keywords (high precision)
                attr_in_keywords = any(v in keywords for v in attr_variations)
                
                if has_person and has_attribute:
                    if attr_in_keywords:
                        score += 15  # Very strong: person + attribute in keywords
                    else:
                        score += 8   # Strong: person + attribute co-occur
                elif has_person:
                    score += 2   # Weaker: only person mentioned
            all_words = set(all_text.split())
            
            # Score based on word overlap with query
            overlap = len(query_words & all_words)
            score += overlap * 2
            
            # Boost if target entity is mentioned
            if target_lower and target_lower in all_text:
                score += 3
            
            scored_nodes.append((score, node))
        
        # Sort by score (descending)
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        
        return [node for score, node in scored_nodes]
    
    def _get_word_variations(self, word: str) -> List[str]:
        """
        [GENERIC] Generate morphological variations of a word.
        No hardcoded synonyms - just linguistic stemming patterns.
        """
        w = word.lower().strip()
        variations = {w}
        
        # Remove common suffixes to get stem
        for suffix in ['s', 'es', 'ed', 'ing', 'tion', 'ness']:
            if w.endswith(suffix) and len(w) > len(suffix) + 2:
                stem = w[:-len(suffix)]
                variations.add(stem)
                # Add other forms of the stem
                variations.add(stem + 's')
                variations.add(stem + 'ing')
        
        # Add common suffixes to original word
        variations.add(w + 's')
        variations.add(w + 'ed')
        variations.add(w + 'ing')
        
        # Remove duplicates and empty strings
        return [v for v in variations if v and len(v) > 1]

    def _hybrid_recall(self, query: str) -> List[MemoryEntry]:
        """
        Hybrid Retrieval - Eq. 7: V_term = Top-k_sem(q) ∪ Top-k_lex(q)
        
        Paper Reference: Section 2.3 - Hybrid Retrieval
        Combines dense (semantic) and sparse (lexical) retrieval
        """
        import config
        
        candidates = {}  # Map[id, entry] to deduplicate
        
        # Pre-extract keywords (fast, no I/O)
        query_lower = query.lower()
        skip_words = {'what', 'when', 'where', 'who', 'why', 'how', 'does', 'did', 'have', 'has', 'the', 'about', 'from', 'with', 'like', 'think', 'are', 'is', 'was', 'were'}
        keywords = [w.strip('?.,!\'') for w in query_lower.split() if len(w) > 2 and w.lower() not in skip_words]
        
        # Extract from possessive pattern
        possessive_match = re.search(r"([A-Za-z]+)'s\s+(\w+)", query)
        if possessive_match:
            attr_word = possessive_match.group(2).lower()
            keywords.extend(self._get_word_variations(attr_word)[:3])
        
        # [PARALLEL] Run semantic + keyword searches concurrently
        def do_semantic():
            try:
                return self.vector_store.semantic_search(query, top_k=config.SEMANTIC_TOP_K) or []
            except: return []
        
        def do_keyword():
            if not keywords:
                return []
            try:
                return self.vector_store.keyword_search(keywords[:5], top_k=config.KEYWORD_TOP_K) or []
            except: return []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_semantic = executor.submit(do_semantic)
            future_keyword = executor.submit(do_keyword)
            
            for res in future_semantic.result():
                if res and res.entry_id:
                    candidates[res.entry_id] = res
            for res in future_keyword.result():
                if res and res.entry_id and res.entry_id not in candidates:
                    candidates[res.entry_id] = res

        return list(candidates.values())

    def _build_inference_graph(self, nodes: List[MemoryEntry]) -> GraphPath:
        """
        Build query-specific evidence graph G_q with bridge discovery
        
        Paper Reference: Section 2.3 - Base Graph Construction & Bridge Discovery
        1. Sort nodes chronologically
        2. Construct edges via entity/temporal overlap (Eq. 8)
        3. For disconnected pairs, search for bridge nodes (Eq. 9)
        4. Mine reasoning paths via DFS (Eq. 10)
        """
        if not nodes:
            return GraphPath([], [])

        final_nodes_map = {n.entry_id: n for n in nodes}
        edges = []
        
        # Step 1: Chronological sort (builds narrative backbone)
        sorted_nodes = sorted(
            nodes, 
            key=lambda x: str(x.timestamp) if x.timestamp else "9999-99-99"
        )
        
        # Step 2: Edge Construction (Eq. 8)
        # Connect nodes via entity overlap or temporal proximity (delta_time < 6 hours)
        for i in range(len(sorted_nodes) - 1):
            curr = sorted_nodes[i]
            next_node = sorted_nodes[i+1]
            
            # Check connectivity via entity/temporal overlap
            conn_type = self._check_connection(curr, next_node)
            
            if conn_type:
                # Case A: Direct connection already exists
                edges.append({
                    'source': curr, 
                    'target': next_node, 
                    'type': conn_type,
                    'info': 'direct'
                })
            else:
                # Case B: Disconnected pair - try bridge discovery (Eq. 9)
                # Paper: Search for bridge node b* when time gap is moderate (1-168 hours)
                time_diff = self._check_time_proximity(curr.timestamp, next_node.timestamp)
                
                # Bridge search window: 1-168 hours (extended for LoCoMo's longer conversations)
                if time_diff and 1.0 < time_diff < 168.0: 
                    # Steiner tree approximation: find bridge node b*
                    bridge_node = self._find_bridge_node(curr, next_node)
                    
                    if bridge_node:
                        # Bridge discovered - add to graph
                        final_nodes_map[bridge_node.entry_id] = bridge_node
                        
                        # Establish path: m_i -> b* -> m_j
                        edges.append({'source': curr, 'target': bridge_node, 'type': 'bridge_in', 'info': 'inferred'})
                        edges.append({'source': bridge_node, 'target': next_node, 'type': 'bridge_out', 'info': 'inferred'})

        # Re-sort with added bridge nodes for topology-aware synthesis
        final_node_list = sorted(
            final_nodes_map.values(),
            key=lambda x: str(x.timestamp) if x.timestamp else "9999"
        )

        # Step 3: Multi-hop path mining via DFS (Eq. 10)
        reasoning_paths = self._discover_reasoning_paths(final_node_list, edges)

        return GraphPath(nodes=final_node_list, edges=edges, reasoning_paths=reasoning_paths)

    def _check_connection(self, n1: MemoryEntry, n2: MemoryEntry) -> Optional[str]:
        """Check direct connectivity between two nodes"""
        # Defensive programming: ensure fields are not None
        k1 = set((n1.keywords or []) + (n1.entities or []) + (n1.persons or []))
        k2 = set((n2.keywords or []) + (n2.entities or []) + (n2.persons or []))
        
        # 1. Entity overlap (strong semantic connection)
        if k1.intersection(k2):
            return "entity_link"
        
        # 2. Temporal overlap (implicit causality)
        # If two events occur within 6 hours, even without co-occurring words, consider it as temporal flow
        dt = self._check_time_proximity(n1.timestamp, n2.timestamp)
        if dt is not None and dt < 6.0:
            return "temporal_flow"
            
        return None

    def _find_bridge_node(self, n1: MemoryEntry, n2: MemoryEntry) -> Optional[MemoryEntry]:
        """
        Algorithmic Bridge Discovery - Eq. 9 in paper
        
        Paper Reference: Section 2.3 - Algorithmic Bridge Discovery
        Finds b* = argmax_{m in V\\V_term} cos(E(q_ij), v_m) s.t. t_m in [t_i, t_j]
        where q_ij = Ent(m_i) ∪ Ent(m_j) ∪ K_i ∪ K_j
        """
        endpoint_ids = {n1.entry_id, n2.entry_id}
        
        # Extract features from both endpoints
        e1 = set((n1.entities or []) + (n1.persons or []))
        e2 = set((n2.entities or []) + (n2.persons or []))
        k1 = set(n1.keywords or [])
        k2 = set(n2.keywords or [])
        
        # Strategy 1: Entity-based bridge (shared entities)
        # If both nodes mention different entities, find a node that connects them
        shared_entities = e1 & e2
        bridge_entities = (e1 | e2) - shared_entities
        
        if bridge_entities:
            bridge_query = ' '.join(list(bridge_entities)[:4])
            candidate = self._search_bridge_candidate(
                bridge_query, endpoint_ids, n1.timestamp, n2.timestamp
            )
            if candidate:
                return candidate
        
        # Strategy 2: Keyword combination (original approach, enhanced)
        all_keywords = list(k1 | k2)[:5]
        all_entities = list(e1 | e2)[:3]
        
        if all_keywords or all_entities:
            bridge_query = ' '.join(all_entities + all_keywords)
            candidate = self._search_bridge_candidate(
                bridge_query, endpoint_ids, n1.timestamp, n2.timestamp
            )
            if candidate:
                return candidate
        
        # Strategy 3: Content-based search using lossless_restatement
        if n1.lossless_restatement and n2.lossless_restatement:
            # Extract key nouns from lossless_restatement
            words1 = [w for w in n1.lossless_restatement.split() if len(w) > 4 and w[0].isupper()][:2]
            words2 = [w for w in n2.lossless_restatement.split() if len(w) > 4 and w[0].isupper()][:2]
            if words1 or words2:
                bridge_query = ' '.join(words1 + words2)
                candidate = self._search_bridge_candidate(
                    bridge_query, endpoint_ids, n1.timestamp, n2.timestamp
                )
                if candidate:
                    return candidate
            
        return None
    
    def _search_bridge_candidate(
        self, 
        query: str, 
        exclude_ids: Set[str],
        ts1: Optional[str],
        ts2: Optional[str]
    ) -> Optional[MemoryEntry]:
        """
        Helper: Search for bridge candidate with time-aware filtering
        """
        try:
            if not hasattr(self.vector_store, 'semantic_search'):
                return None
                
            candidates = self.vector_store.semantic_search(query, top_k=5)
            
            for candidate in candidates:
                if not candidate or candidate.entry_id in exclude_ids:
                    continue
                
                # Time-aware filtering: bridge should be temporally between endpoints
                if ts1 and ts2 and candidate.timestamp:
                    ts_bridge = candidate.timestamp
                    # Check if bridge is between the two endpoints
                    if self._is_timestamp_between(ts_bridge, ts1, ts2):
                        return candidate
                    # If time check fails, still accept if it's close to either endpoint
                    dt1 = self._check_time_proximity(ts_bridge, ts1)
                    dt2 = self._check_time_proximity(ts_bridge, ts2)
                    if dt1 and dt2 and (dt1 < 24 or dt2 < 24):
                        return candidate
                else:
                    # No timestamp info, just return first valid candidate
                    return candidate
                    
        except Exception:
            pass
            
        return None
    
    def _is_timestamp_between(self, ts: str, ts1: str, ts2: str) -> bool:
        """Check if timestamp ts is between ts1 and ts2"""
        try:
            t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(str(ts1).replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(str(ts2).replace("Z", "+00:00"))
            
            # Normalize timezone
            if t.tzinfo is None: t = t.replace(tzinfo=None)
            if t1.tzinfo is None: t1 = t1.replace(tzinfo=None)
            if t2.tzinfo is None: t2 = t2.replace(tzinfo=None)
            if t.tzinfo is not None: t = t.replace(tzinfo=None)
            if t1.tzinfo is not None: t1 = t1.replace(tzinfo=None)
            if t2.tzinfo is not None: t2 = t2.replace(tzinfo=None)
            
            min_t, max_t = (t1, t2) if t1 <= t2 else (t2, t1)
            return min_t <= t <= max_t
        except:
            return False

    def _discover_reasoning_paths(self, nodes: List[MemoryEntry], edges: List[Dict[str, Any]]) -> List[List[MemoryEntry]]:
        """
        Multi-Hop Path Mining via DFS - Eq. 10 in paper
        
        Paper Reference: Section 2.3 - Multi-Hop Path Mining & Node Budget
        P_q = {p in G_q | 2 <= |p| <= L, temporally consistent}
        where L = 3 hops (max depth)
        
        Returns paths for topology-aware synthesis (guides LLM reasoning)
        """
        if not nodes or not edges:
            return []
        
        # Build adjacency list for DFS traversal
        adj = defaultdict(list)
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target and hasattr(source, 'entry_id') and hasattr(target, 'entry_id'):
                adj[source.entry_id].append(target)
        
        # DFS to discover all paths (L = max depth from config)
        max_path_depth = getattr(config, 'MAX_REASONING_PATH_DEPTH', 3)
        all_paths = []
        
        def dfs(current_node: MemoryEntry, path: List[MemoryEntry], visited: Set[str], max_depth: int = max_path_depth):
            """DFS to discover reasoning paths"""
            # Stop condition: reached max depth
            if len(path) >= max_depth:
                if len(path) >= 2:  # At least 2 nodes to form a path
                    all_paths.append(path[:])
                return
            
            # Check if current node has any neighbors to explore
            neighbors = adj.get(current_node.entry_id, [])
            has_unvisited_neighbor = any(n.entry_id not in visited for n in neighbors)
            
            # If no more neighbors to explore, save current path (leaf node)
            if not has_unvisited_neighbor and len(path) >= 2:
                all_paths.append(path[:])
                return
            
            # Explore neighbors
            for next_node in neighbors:
                if next_node.entry_id not in visited:
                    visited.add(next_node.entry_id)
                    path.append(next_node)
                    dfs(next_node, path, visited, max_depth)
                    path.pop()
                    visited.remove(next_node.entry_id)
        
        # Start DFS from each node
        for node in nodes:
            visited = {node.entry_id}
            dfs(node, [node], visited, max_depth=max_path_depth)
        
        # Deduplicate and rank paths by length (prefer longer chains)
        unique_paths = []
        seen = set()
        
        # Sort by length (prefer longer chains) and then by time
        sorted_paths = sorted(
            all_paths,
            key=lambda p: (
                len(p),  # Longer paths first
                min([n.timestamp or "9999" for n in p])  # Earlier paths first
            ),
            reverse=True
        )
        
        for path in sorted_paths:
            path_id = tuple(n.entry_id for n in path)
            if path_id not in seen:
                unique_paths.append(path)
                seen.add(path_id)
        
        # Return top paths (scale with graph size, cap at config limit)
        max_paths_limit = getattr(config, 'MAX_REASONING_PATHS', 10)
        max_paths = min(len(nodes), max_paths_limit)
        return unique_paths[:max_paths]

    def _check_time_proximity(self, ts1: Optional[str], ts2: Optional[str]) -> Optional[float]:
        """
        Helper: Return hours between timestamps, or None.
        Handles formatting robustly.
        """
        if not ts1 or not ts2: return None
        try:
            # Handle ISO format, remove Z to prevent python version compatibility issues
            t1_str = str(ts1).replace("Z", "+00:00")
            t2_str = str(ts2).replace("Z", "+00:00")
            
            t1 = datetime.fromisoformat(t1_str)
            t2 = datetime.fromisoformat(t2_str)
            
            # Handle timezone (Naive vs Aware) - unify to timezone-naive or unified timezone
            if t1.tzinfo is None and t2.tzinfo is not None: t2 = t2.replace(tzinfo=None)
            if t1.tzinfo is not None and t2.tzinfo is None: t1 = t1.replace(tzinfo=None)
            
            return abs((t1 - t2).total_seconds()) / 3600.0
        except:
            return None