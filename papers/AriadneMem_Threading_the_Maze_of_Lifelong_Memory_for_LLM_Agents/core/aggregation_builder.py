"""
Aggregation Cache Builder - Generic entity and relation aggregation
Automatically learns patterns from memory entries without hard-coded rules
"""
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re
from datetime import datetime

from models.memory_entry import MemoryEntry
from models.enhanced_structures import (
    EntityAggregation, RelationTriple, EnhancedMemoryIndex, TemporalInfo
)


class AggregationBuilder:
    """
    Builds aggregated views of memory without domain-specific templates
    Uses NLP patterns to automatically discover:
    - Entity attributes and activities
    - Relation triples (subject-predicate-object)
    - Temporal sequences and counts
    """
    
    def __init__(self):
        # Common action verbs (extensible, learned from data)
        self.action_verbs = {
            'went', 'visited', 'traveled', 'painted', 'read', 'watched',
            'bought', 'sold', 'made', 'created', 'attended', 'played',
            'adopted', 'rejected', 'wrote', 'drew', 'designed', 'built',
            'participated', 'joined', 'left', 'started', 'finished',
            'likes', 'enjoys', 'prefers', 'loves', 'hates', 'dislikes'
        }
        
        # Dynamically expanded during processing
        self.discovered_actions = set()
        self.discovered_attributes = defaultdict(set)
    
    def build_aggregations(self, entries: List[MemoryEntry]) -> EnhancedMemoryIndex:
        """
        Build all aggregations from memory entries
        Returns: EnhancedMemoryIndex with entities, relations, temporal index
        """
        index = EnhancedMemoryIndex()
        
        # Group entries by entity
        entity_entries = defaultdict(list)
        
        for entry in entries:
            # Extract entities from persons and entities fields
            all_entities = set()
            all_entities.update(entry.persons)
            all_entities.update(entry.entities)
            
            for entity in all_entities:
                if entity:  # Skip empty strings
                    entity_entries[entity].append(entry)
        
        # Build entity aggregations
        for entity_name, entity_entry_list in entity_entries.items():
            aggregation = self._aggregate_entity(entity_name, entity_entry_list)
            index.entities[entity_name] = aggregation
        
        # Build relation triples
        for entry in entries:
            triples = self._extract_relations(entry)
            index.relations.extend(triples)
        
        # Build temporal index
        index.temporal_index = self._build_temporal_index(entries)
        
        return index
    
    def _aggregate_entity(self, entity_name: str, entries: List[MemoryEntry]) -> EntityAggregation:
        """
        Aggregate all information about a single entity
        Uses pattern matching to discover attributes and events
        """
        # Determine entity type
        entity_type = self._infer_entity_type(entity_name, entries)
        
        # Initialize aggregation
        agg = EntityAggregation(
            entity_name=entity_name,
            entity_type=entity_type,
            evidence_entries=[e.entry_id for e in entries]
        )
        
        # Extract and count events
        for entry in entries:
            text = entry.lossless_restatement.lower()
            
            # Pattern 1: "entity verb something" (action detection)
            actions = self._extract_actions(entity_name, text)
            for action in actions:
                agg.event_counts[action] = agg.event_counts.get(action, 0) + 1
            
            # Pattern 2: "entity has/owns/likes X" (attribute detection)
            attributes = self._extract_attributes(entity_name, entry)
            for attr_type, values in attributes.items():
                if attr_type not in agg.attribute_sets:
                    agg.attribute_sets[attr_type] = set()
                agg.attribute_sets[attr_type].update(values)
            
            # Pattern 3: Temporal sequences (first, last, count)
            if entry.timestamp:
                for action in actions:
                    if action not in agg.temporal_sequences:
                        agg.temporal_sequences[action] = (entry.timestamp, entry.timestamp, 1)
                    else:
                        first, last, count = agg.temporal_sequences[action]
                        agg.temporal_sequences[action] = (
                            min(first, entry.timestamp),
                            max(last, entry.timestamp),
                            count + 1
                        )
        
        return agg
    
    def _infer_entity_type(self, entity_name: str, entries: List[MemoryEntry]) -> str:
        """
        Infer entity type from name and context
        """
        # Check if entity is in persons list (most common case)
        for entry in entries:
            if entity_name in entry.persons:
                return "person"
        
        # Check location indicators
        location_keywords = ['city', 'country', 'place', 'location', 'street', 'building']
        for entry in entries:
            if entry.location and entity_name.lower() in entry.location.lower():
                return "location"
        
        # Default to concept/entity
        return "entity"
    
    def _extract_actions(self, entity_name: str, text: str) -> List[str]:
        """
        Extract actions performed by or related to entity
        Pattern: "entity [action_verb] [object/location]"
        """
        actions = []
        
        # Normalize entity name for matching
        entity_lower = entity_name.lower()
        
        # Find entity mentions
        if entity_lower not in text:
            return actions
        
        # Split into sentences
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if entity_lower not in sentence:
                continue
            
            # Check for known action verbs
            for verb in self.action_verbs:
                if verb in sentence:
                    # Extract action phrase
                    action = self._extract_action_phrase(sentence, verb, entity_lower)
                    if action:
                        actions.append(action)
                        self.discovered_actions.add(action)
        
        return actions
    
    def _extract_action_phrase(self, sentence: str, verb: str, entity: str) -> str:
        """
        Extract meaningful action phrase from sentence
        Returns: "verb_object" (e.g., "visited_beach", "painted_sunset")
        """
        # Find verb position
        verb_pos = sentence.find(verb)
        if verb_pos == -1:
            return None
        
        # Check if entity is subject (before verb)
        entity_pos = sentence.find(entity)
        if entity_pos == -1 or entity_pos > verb_pos:
            return None
        
        # Extract object after verb
        after_verb = sentence[verb_pos + len(verb):].strip()
        
        # Get first meaningful noun phrase (up to 3 words)
        words = after_verb.split()[:3]
        
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'to', 'at', 'in', 'on', 'for', 'with', 'and', 'or'}
        obj_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        if obj_words:
            obj = '_'.join(obj_words[:2])  # Max 2 words
            return f"{verb}_{obj}"
        else:
            return verb  # Just the verb if no clear object
    
    def _extract_attributes(self, entity_name: str, entry: MemoryEntry) -> Dict[str, Set[str]]:
        """
        Extract entity attributes from memory entry
        Uses keywords, topics, and pattern matching
        """
        attributes = defaultdict(set)
        
        # Use existing structured fields
        if entry.keywords:
            # Filter keywords related to entity
            entity_keywords = [kw for kw in entry.keywords if entity_name.lower() not in kw.lower()]
            if entity_keywords:
                attributes['keywords'].update(entity_keywords)
        
        if entry.topic and entity_name.lower() in entry.lossless_restatement.lower():
            attributes['topics'].add(entry.topic)
        
        if entry.location:
            attributes['locations'].add(entry.location)
        
        # Extract from lossless_restatement
        text = entry.lossless_restatement.lower()
        entity_lower = entity_name.lower()
        
        if entity_lower in text:
            # Pattern: "entity's X" (possessive)
            possessive_pattern = rf"{re.escape(entity_lower)}'s\s+(\w+(?:\s+\w+)?)"
            matches = re.findall(possessive_pattern, text)
            if matches:
                attributes['possessions'].update(matches)
            
            # Pattern: "entity has/owns/likes X"
            has_pattern = rf"{re.escape(entity_lower)}\s+(?:has|owns|likes|prefers|enjoys)\s+([^,.;]+)"
            matches = re.findall(has_pattern, text)
            if matches:
                for match in matches:
                    clean_val = match.strip()[:50]  # Limit length
                    if clean_val:
                        attributes['preferences'].add(clean_val)
        
        return attributes
    
    def _extract_relations(self, entry: MemoryEntry) -> List[RelationTriple]:
        """
        Extract relation triples from memory entry
        Pattern: (subject, predicate, object)
        """
        triples = []
        text = entry.lossless_restatement
        
        # Use persons and entities as potential subjects/objects
        entities = list(set(entry.persons + entry.entities))
        
        if len(entities) < 2:
            # Need at least 2 entities for a relation
            return triples
        
        # For each pair of entities, check if there's a relation
        for i, subj in enumerate(entities):
            for obj in entities[i+1:]:
                # Check if both appear in the statement
                if subj.lower() in text.lower() and obj.lower() in text.lower():
                    # Find connecting verb
                    predicate = self._find_connecting_verb(text, subj, obj)
                    if predicate:
                        triple = RelationTriple(
                            subject=subj,
                            predicate=predicate,
                            object=obj,
                            timestamp=entry.timestamp,
                            location=entry.location,
                            source_entry_id=entry.entry_id
                        )
                        triples.append(triple)
        
        return triples
    
    def _find_connecting_verb(self, text: str, subj: str, obj: str) -> str:
        """
        Find verb that connects two entities in text
        """
        text_lower = text.lower()
        subj_pos = text_lower.find(subj.lower())
        obj_pos = text_lower.find(obj.lower())
        
        if subj_pos == -1 or obj_pos == -1:
            return None
        
        # Get text between entities
        start = min(subj_pos, obj_pos)
        end = max(subj_pos, obj_pos)
        between = text_lower[start:end]
        
        # Find verb in between
        for verb in self.action_verbs:
            if verb in between:
                return verb
        
        # Check for common relational phrases
        relation_phrases = ['with', 'and', 'along with', 'together with', 'both']
        for phrase in relation_phrases:
            if phrase in between:
                return phrase
        
        return "related_to"  # Default relation
    
    def _build_temporal_index(self, entries: List[MemoryEntry]) -> Dict[str, List[str]]:
        """
        Build index from dates to entry IDs for fast temporal queries
        """
        temporal_index = defaultdict(list)
        
        for entry in entries:
            if entry.timestamp:
                # Extract date part (YYYY-MM-DD)
                try:
                    date_str = entry.timestamp[:10]  # Get YYYY-MM-DD
                    temporal_index[date_str].append(entry.entry_id)
                except:
                    pass  # Skip invalid timestamps
        
        return dict(temporal_index)
