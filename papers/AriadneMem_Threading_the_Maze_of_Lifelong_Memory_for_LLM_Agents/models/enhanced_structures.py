"""
Enhanced Data Structures for AriadneMem
Provides general-purpose caching and indexing without hard-coded domain logic

Paper Reference: Section 2.3 - Fast Paths and Enhanced Caching
"""
from typing import List, Dict, Any, Optional, Set, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
from collections import defaultdict
import json


class TemporalInfo(BaseModel):
    """
    Structured temporal information with flexible precision
    Supports both absolute and relative time expressions
    """
    # Original string representation
    raw_expression: str
    
    # Parsed datetime (if parseable)
    parsed_datetime: Optional[str] = None  # ISO format
    precision: Optional[str] = None  # "day", "month", "year", "hour"
    
    # Relative expressions (kept as-is for semantic matching)
    is_relative: bool = False
    relative_anchor: Optional[str] = None  # e.g., "before July 6"
    
    # Fuzzy time ranges (for interval queries)
    start_datetime: Optional[str] = None
    end_datetime: Optional[str] = None


class EntityAggregation(BaseModel):
    """
    Generic entity-level aggregation without predefined categories
    Automatically learns patterns from data
    """
    entity_name: str
    entity_type: str  # "person", "location", "concept", etc.
    
    # Count-based aggregations (learned from data)
    # Key: action/event type, Value: count
    event_counts: Dict[str, int] = Field(default_factory=dict)
    
    # Set-based aggregations (complete lists)
    # Key: attribute type, Value: set of values
    attribute_sets: Dict[str, Set[str]] = Field(default_factory=dict)
    
    # Temporal aggregations (first/last occurrences)
    # Key: event type, Value: (first_time, last_time, count)
    temporal_sequences: Dict[str, Tuple[str, str, int]] = Field(default_factory=dict)
    
    # Supporting memory entry IDs
    evidence_entries: List[str] = Field(default_factory=list)
    
    class Config:
        # Allow sets to be JSON serializable
        json_encoders = {
            set: list
        }


class RelationTriple(BaseModel):
    """
    Generic relation representation (subject, predicate, object)
    Extracted automatically from lossless_restatement
    """
    subject: str
    predicate: str  # Action/relationship verb
    object: str
    
    # Context
    timestamp: Optional[str] = None
    location: Optional[str] = None
    
    # Evidence
    source_entry_id: str
    confidence: float = 1.0  # Can be used for fuzzy matching


class QueryCache(BaseModel):
    """
    Cache for frequently accessed query patterns
    Learns common access patterns without hard-coding
    """
    cache_key: str  # Hash of query pattern
    
    # Cached results (flexible structure)
    cached_value: Any
    value_type: str  # "count", "list", "entity", "relation"
    
    # Metadata
    hit_count: int = 0
    last_accessed: Optional[str] = None
    
    # Supporting evidence
    source_entries: List[str] = Field(default_factory=list)


class SemanticNormalizationRule(BaseModel):
    """
    Learned normalization patterns (not pre-defined synonyms)
    Built from reference-answer pairs dynamically
    """
    pattern_type: str  # "synonym", "format", "ordering", "granularity"
    
    # Pattern matching (flexible)
    source_pattern: str
    target_pattern: str
    
    # Context (when this rule applies)
    context_keywords: List[str] = Field(default_factory=list)
    
    # Confidence
    usage_count: int = 0
    success_rate: float = 1.0


class EnhancedMemoryIndex(BaseModel):
    """
    Container for all enhanced indexing structures
    All components are optional and built on-demand
    """
    # Entity-level aggregations
    entities: Dict[str, EntityAggregation] = Field(default_factory=dict)
    
    # Relation triples (for relationship queries)
    relations: List[RelationTriple] = Field(default_factory=list)
    
    # Query cache (for repeated patterns)
    query_cache: Dict[str, QueryCache] = Field(default_factory=dict)
    
    # Temporal index (for time-range queries)
    temporal_index: Dict[str, List[str]] = Field(default_factory=dict)  # date -> entry_ids
    
    # Semantic normalization (learned rules)
    normalization_rules: List[SemanticNormalizationRule] = Field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "entities": {k: v.dict() for k, v in self.entities.items()},
            "relations": [r.dict() for r in self.relations],
            "query_cache": {k: v.dict() for k, v in self.query_cache.items()},
            "temporal_index": self.temporal_index,
            "normalization_rules": [r.dict() for r in self.normalization_rules]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EnhancedMemoryIndex":
        """Load from dictionary"""
        # Handle sets in attribute_sets
        entities = {}
        for k, v in data.get("entities", {}).items():
            # Convert lists back to sets for attribute_sets
            if "attribute_sets" in v:
                v["attribute_sets"] = {
                    attr_k: set(attr_v) if isinstance(attr_v, list) else attr_v
                    for attr_k, attr_v in v["attribute_sets"].items()
                }
            entities[k] = EntityAggregation(**v)
        
        return cls(
            entities=entities,
            relations=[RelationTriple(**r) for r in data.get("relations", [])],
            query_cache={k: QueryCache(**v) for k, v in data.get("query_cache", {}).items()},
            temporal_index=data.get("temporal_index", {}),
            normalization_rules=[SemanticNormalizationRule(**r) for r in data.get("normalization_rules", [])]
        )
