"""
Vector Store - Structured Multi-View Indexing Implementation (Section 3.2)

Paper Reference: Section 3.2 - Structured Indexing
Implements the three structured indexing dimensions:
- Semantic Layer: Dense vectors v_k ∈ ℝ^d (embedding-based similarity)
- Lexical Layer: Sparse vectors h_k ∈ ℝ^|V| (BM25/keyword matching)
- Symbolic Layer: Metadata R_k = {(key, val)} (structured filtering by time, entities, etc.)
"""
from typing import List, Optional, Dict, Any, Union
import lancedb
import pyarrow as pa
import numpy as np
import json
from pathlib import Path
from models.memory_entry import MemoryEntry
from models.enhanced_structures import EnhancedMemoryIndex
from utils.embedding import EmbeddingModel
import config
import os


def _safe_str(value: Any) -> Optional[str]:
    """Convert value to string safely, handling lists and None."""
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else None
    if isinstance(value, str):
        return value if value else None
    return str(value)


def _safe_list(value: Any) -> List[str]:
    """Convert value to list of strings safely."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value] if value else []
    return [str(value)]


def _safe_memory_entry(data: Dict[str, Any]) -> MemoryEntry:
    """Create MemoryEntry with type-safe field conversion."""
    return MemoryEntry(
        entry_id=str(data.get("entry_id", "")),
        lossless_restatement=str(data.get("lossless_restatement", "")),
        keywords=_safe_list(data.get("keywords")),
        timestamp=_safe_str(data.get("timestamp")),
        location=_safe_str(data.get("location")),
        persons=_safe_list(data.get("persons")),
        entities=_safe_list(data.get("entities")),
        topic=_safe_str(data.get("topic"))
    )


class VectorStore:
    """
    Structured Multi-View Indexing - Storage and retrieval for Atomic Entries

    Paper Reference: Section 3.2 - Structured Indexing
    Implements M(m_k) with three structured layers:
    1. Semantic Layer: Dense embedding vectors for conceptual similarity
    2. Lexical Layer: Sparse keyword vectors for precise term matching
    3. Symbolic Layer: Structured metadata for deterministic filtering
    """
    def __init__(self, db_path: str = None, embedding_model: EmbeddingModel = None, table_name: str = None):
        self.db_path = db_path or config.LANCEDB_PATH
        self.embedding_model = embedding_model or EmbeddingModel()

        # Connect to database
        os.makedirs(self.db_path, exist_ok=True)
        self.db = lancedb.connect(self.db_path)
        self.table_name = table_name or config.MEMORY_TABLE_NAME
        self.table = None

        # Enhanced index storage path
        self.enhanced_index_path = os.path.join(self.db_path, "enhanced_index.json")
        self.enhanced_index = None

        self._init_table()

    def _init_table(self):
        """
        Initialize table schema
        """
        # Define schema
        schema = pa.schema([
            pa.field("entry_id", pa.string()),
            pa.field("lossless_restatement", pa.string()),
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("timestamp", pa.string()),
            pa.field("location", pa.string()),
            pa.field("persons", pa.list_(pa.string())),
            pa.field("entities", pa.list_(pa.string())),
            pa.field("topic", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), self.embedding_model.dimension))
        ])

        # Create table if it doesn't exist
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"Created new table: {self.table_name}")
        else:
            self.table = self.db.open_table(self.table_name)
            print(f"Opened existing table: {self.table_name}")

    def add_entries(self, entries: List[MemoryEntry]):
        """
        Batch add memory entries
        """
        if not entries:
            return

        # Generate vectors (encode documents without query prompt)
        restatements = [entry.lossless_restatement for entry in entries]
        vectors = self.embedding_model.encode_documents(restatements)

        # Build data
        data = []
        for entry, vector in zip(entries, vectors):
            data.append({
                "entry_id": entry.entry_id,
                "lossless_restatement": entry.lossless_restatement,
                "keywords": entry.keywords,
                "timestamp": entry.timestamp or "",
                "location": entry.location or "",
                "persons": entry.persons,
                "entities": entry.entities,
                "topic": entry.topic or "",
                "vector": vector.tolist()
            })

        # Add to table
        self.table.add(data)
        print(f"Added {len(entries)} memory entries")

    def semantic_search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """
        Semantic Layer Search - Dense vector similarity

        Paper Reference: Section 3.1
        Retrieves based on v_k = E_dense(S_k) where S_k is the lossless restatement
        """
        try:
            # Generate query vector (use query prompt optimization for Qwen3)
            query_vector = self.embedding_model.encode_single(query, is_query=True)

            # Execute vector search (LanceDB handles empty table gracefully)
            results = self.table.search(query_vector.tolist()).limit(top_k).to_list()

            # Convert to MemoryEntry objects using type-safe helper
            entries = []
            for result in results:
                try:
                    entry = _safe_memory_entry(result)
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Failed to parse search result: {e}")
                    continue

            return entries

        except Exception as e:
            print(f"Error during semantic search: {e}")
            return []

    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """
        [OPTIMIZED] Lexical Layer Search using LanceDB SQL filtering.
        
        Uses native WHERE clause instead of loading entire table.
        O(log n) with index vs O(n) full scan.
        """
        try:
            if not keywords:
                return []
            
            # Build SQL WHERE clause for keyword matching
            # Match against lossless_restatement text
            conditions = []
            for kw in keywords[:5]:  # Limit to 5 keywords
                kw_escaped = kw.replace("'", "''")  # Escape single quotes
                conditions.append(f"lower(lossless_restatement) LIKE '%{kw_escaped.lower()}%'")
            
            if not conditions:
                return []
            
            where_clause = " OR ".join(conditions)
            
            # Use LanceDB's native SQL filtering (much faster than pandas)
            results = self.table.search().where(where_clause).limit(top_k * 3).to_list()
            
            if not results:
                return []
            
            # Score and rank results
            scored_entries = []
            for result in results:
                try:
                    score = 0
                    row_text = str(result["lossless_restatement"]).lower()
                    row_keywords = list(result["keywords"]) if result.get("keywords") else []
                    
                    for kw in keywords:
                        kw_lower = kw.lower()
                        if row_keywords and any(kw_lower in str(rk).lower() for rk in row_keywords):
                            score += 2
                        if kw_lower in row_text:
                            score += 1
                    
                    if score > 0:
                        entry = _safe_memory_entry(result)
                        scored_entries.append((score, entry))
                except Exception as e:
                    continue
            
            scored_entries.sort(reverse=True, key=lambda x: x[0])
            return [entry for _, entry in scored_entries[:top_k]]

        except Exception as e:
            print(f"Error during keyword search: {e}")
            # Fallback to original method if SQL fails
            return self._keyword_search_fallback(keywords, top_k)
    
    def _keyword_search_fallback(self, keywords: List[str], top_k: int = 3) -> List[MemoryEntry]:
        """Fallback method using pandas (slower but more compatible)"""
        try:
            all_entries = self.table.to_pandas()
            if len(all_entries) == 0 or not keywords:
                return []
            
            scored_entries = []
            for _, row in all_entries.iterrows():
                score = 0
                row_keywords = list(row["keywords"]) if row["keywords"] is not None else []
                row_text = str(row["lossless_restatement"]).lower()
                
                for kw in keywords:
                    kw_lower = str(kw).lower()
                    if row_keywords and any(kw_lower in str(rk).lower() for rk in row_keywords):
                        score += 2
                    if kw_lower in row_text:
                        score += 1
                
                if score > 0:
                    entry = _safe_memory_entry(row.to_dict())
                    scored_entries.append((score, entry))
            
            scored_entries.sort(reverse=True, key=lambda x: x[0])
            return [entry for _, entry in scored_entries[:top_k]]
        except:
            return []

    def structured_search(
        self,
        persons: Optional[List[str]] = None,
        timestamp_range: Optional[tuple] = None,
        location: Optional[str] = None,
        entities: Optional[List[str]] = None,
        top_k: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Symbolic Layer Search - Metadata-based deterministic filtering

        Paper Reference: Section 3.1
        Retrieves based on R_k = {(key, val)} for structured constraints
        Enables precise filtering by time, entities, persons, and locations

        Args:
            persons: Filter by person names
            timestamp_range: Filter by time range (start, end)
            location: Filter by location
            entities: Filter by entities
            top_k: Maximum number of results to return (default: no limit)
        """
        try:
            df = self.table.to_pandas()

            # Handle empty dataframe
            if len(df) == 0:
                return []

            # If no filters provided, return empty
            if not any([persons, timestamp_range, location, entities]):
                return []

            # Apply filters using numpy array for proper pandas boolean indexing
            mask = np.ones(len(df), dtype=bool)

            if persons:
                person_mask = np.array([
                    any(p in list(row["persons"]) for p in persons) if row["persons"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & person_mask

            if location:
                location_mask = np.array([
                    location.lower() in str(row["location"]).lower() if row["location"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & location_mask

            if entities:
                entity_mask = np.array([
                    any(e in list(row["entities"]) for e in entities) if row["entities"] is not None else False
                    for _, row in df.iterrows()
                ])
                mask = mask & entity_mask

            if timestamp_range:
                start_time, end_time = timestamp_range
                timestamp_mask = np.array([
                    bool(row["timestamp"] and start_time <= row["timestamp"] <= end_time)
                    for _, row in df.iterrows()
                ])
                mask = mask & timestamp_mask

            # Build results - use numpy boolean array for filtering
            filtered_df = df[mask]

            # Limit results if top_k is specified
            if top_k is not None and len(filtered_df) > top_k:
                filtered_df = filtered_df.head(top_k)

            entries = []
            for _, row in filtered_df.iterrows():
                try:
                    entry = _safe_memory_entry(row.to_dict())
                    entries.append(entry)
                except Exception as e:
                    print(f"Warning: Failed to parse filtered row: {e}")
                    continue

            return entries

        except Exception as e:
            print(f"Error during structured search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_all_entries(self) -> List[MemoryEntry]:
        """
        Get all memory entries
        """
        df = self.table.to_pandas()
        entries = []
        for _, row in df.iterrows():
            try:
                entry = _safe_memory_entry(row.to_dict())
                entries.append(entry)
            except Exception as e:
                print(f"Warning: Failed to parse entry: {e}")
                continue
        return entries

    def clear(self):
        """
        Clear all data including enhanced index
        """
        self.db.drop_table(self.table_name)
        self._init_table()
        
        # Clear enhanced index
        if os.path.exists(self.enhanced_index_path):
            os.remove(self.enhanced_index_path)
        self.enhanced_index = None
        
        print("Database and enhanced index cleared")
    
    def save_enhanced_index(self, index: EnhancedMemoryIndex):
        """Save enhanced index to disk"""
        try:
            with open(self.enhanced_index_path, 'w') as f:
                json.dump(index.to_dict(), f, indent=2)
            self.enhanced_index = index
            print(f"Enhanced index saved ({len(index.entities)} entities, {len(index.relations)} relations)")
        except Exception as e:
            print(f"Warning: Failed to save enhanced index: {e}")
    
    def load_enhanced_index(self) -> Optional[EnhancedMemoryIndex]:
        """Load enhanced index from disk"""
        if self.enhanced_index is not None:
            return self.enhanced_index
        
        if not os.path.exists(self.enhanced_index_path):
            return None
        
        try:
            with open(self.enhanced_index_path, 'r') as f:
                data = json.load(f)
            self.enhanced_index = EnhancedMemoryIndex.from_dict(data)
            print(f"Enhanced index loaded ({len(self.enhanced_index.entities)} entities, {len(self.enhanced_index.relations)} relations)")
            return self.enhanced_index
        except Exception as e:
            print(f"Warning: Failed to load enhanced index: {e}")
            return None

