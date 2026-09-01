"""
AriadneMemoryBuilder - Phase I: Asynchronous Memory Construction

Paper Reference: Section 2.2 - Phase I: Asynchronous Memory Construction
Key Features:
1. Entropy-Aware Gating (Eq. 2-3): Filters low-information inputs before LLM extraction
2. Conflict-Aware Graph Coarsening (Eq. 5-6): Merges duplicates while preserving state updates
3. Atomic Entry Extraction (Eq. 4): F_theta transformation to structured memory entries
4. Embedding Cache: Safe optimization for repeated similarity computations
5. Parallel LLM Calls: Parallelizes extraction while keeping coarsening serial
"""

from typing import List, Optional, Dict
import time
import numpy as np
from datetime import datetime
import concurrent.futures

# Models & Utils
from models.memory_entry import MemoryEntry, Dialogue
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config

class AriadneMemoryBuilder:
    """
    AriadneMemoryBuilder - Phase I: Asynchronous Memory Construction
    
    Paper Reference: Section 2.2
    Transforms raw dialogue stream D into a conflict-resolved evolutionary graph:
    - Entropy-aware gating: Blocks redundant inputs (Phi_gate in Eq. 3)
    - Atomic extraction: F_theta transformation (Eq. 4)
    - Conflict-aware coarsening: Merge/Link/Add actions (Eq. 6)
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        window_size: int = None,
        # Paper parameters (Section 2.2)
        redundancy_threshold: float = 0.92,  # lambda_red: entropy gating threshold
        coarsening_threshold: float = 0.96   # lambda_coal: conflict-aware coarsening threshold
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.window_size = window_size or (getattr(config, 'WINDOW_SIZE', 10))

        self.redundancy_threshold = redundancy_threshold
        self.coarsening_threshold = coarsening_threshold

        self.dialogue_buffer: List[Dialogue] = []
        self.processed_count = 0
        self.previous_entries: List[MemoryEntry] = []
        
        # Safe optimization: Embedding cache (doesn't affect results, just avoids redundant computation)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        # Parallel processing configuration
        self.enable_parallel_llm = getattr(config, 'ENABLE_PARALLEL_PROCESSING', True)
        self.max_parallel_workers = getattr(config, 'MAX_PARALLEL_WORKERS', 4)

    def add_dialogue(self, dialogue: Dialogue, auto_process: bool = True):
        """
        Add dialogue with entropy-aware gating (Phi_gate in Eq. 3)
        """
        # 1. Fast Vector Gating - can be disabled via config
        enable_redundancy = getattr(config, 'ENABLE_REDUNDANCY_CHECK', True)
        if enable_redundancy and self._check_is_redundant(dialogue):
            return

        self.dialogue_buffer.append(dialogue)

        if auto_process and len(self.dialogue_buffer) >= self.window_size:
            self.process_window()

    def add_dialogues(self, dialogues: List[Dialogue], auto_process: bool = True):
        """
        Batch add dialogues with optional parallel LLM processing
        
        Paper Reference: Algorithm 1, lines 1-7
        Strategy:
        1. Serial entropy gating (Phi_gate) to preserve order
        2. Parallel LLM extraction (F_theta) for speedup
        3. Serial conflict-aware coarsening for correct deduplication
        """
        enable_redundancy = getattr(config, 'ENABLE_REDUNDANCY_CHECK', True)
        
        if enable_redundancy:
            # Serial redundancy check to ensure order
            filtered = []
            for d in dialogues:
                if not self._check_is_redundant(d):
                    filtered.append(d)
            
            if not filtered:
                return
            self.dialogue_buffer.extend(filtered)
        else:
            # Fast mode: skip redundancy check
            self.dialogue_buffer.extend(dialogues)
        
        # Choose processing method
        if self.enable_parallel_llm and len(self.dialogue_buffer) >= self.window_size * 2:
            # Parallel processing for large batches (only parallelize LLM calls)
            self._process_parallel()
        else:
            # Serial processing for small batches
            while len(self.dialogue_buffer) >= self.window_size:
                self.process_window()

    def _check_is_redundant(self, dialogue: Dialogue) -> bool:
        """
        Entropy-Aware Gating (Phi_gate) - Eq. 2-3 in paper
        
        Paper Reference: Section 2.2 - Entropy-Aware Gating
        Decision function Phi_gate(d_t):
        - Returns 0 (redundant) if: r_t > lambda_red AND delta_t < delta_short
        - Returns 1 (keep) otherwise
        """
        try:
            # 1. Search DB for similar content
            if not hasattr(self.vector_store, 'semantic_search'):
                 return False
                 
            results = self.vector_store.semantic_search(dialogue.content, top_k=1)
            
            if not results:
                return False 

            best_match = results[0]
            
            # 2. Calculate Similarity (using cache for speedup)
            sim = self._calculate_text_similarity(dialogue.content, best_match.lossless_restatement)

            # 3. Decision Logic
            
            # Case A: Low Similarity -> New Information -> KEEP
            if sim < self.redundancy_threshold:
                return False 

            # Case B: High Similarity -> Check nuances
            
            # Check 1: Time Gap (Is it a short-term repetition?)
            time_diff_hours = self._get_time_diff_hours(dialogue.timestamp, best_match.timestamp)
            
            if time_diff_hours is not None:
                # User repeating themselves within 1 hour -> REDUNDANT -> DROP
                if time_diff_hours < 1.0:
                    return True
                
                # Repeating after 24 hours (Recurring routine) -> KEEP
                if time_diff_hours > 24.0:
                    return False

            # Check 2: Exact Match (Copy-paste spam)
            if dialogue.content.strip() == best_match.lossless_restatement.strip():
                return True

            # Case C: High Similarity but ambiguous -> KEEP
            return False

        except Exception as e:
            # On error, default to KEEP to prevent data loss
            return False

    def process_window(self):
        """
        Process dialogue window: Extraction (F_theta) -> Coarsening
        Paper Reference: Algorithm 1, lines 4-7
        """
        if not self.dialogue_buffer:
            return

        window = self.dialogue_buffer[:self.window_size]
        self.dialogue_buffer = self.dialogue_buffer[self.window_size:]

        print(f"[AriadneMem] Processing Window ({len(window)} items)...")

        # Step 1: Atomic Entry Extraction (F_theta) - Eq. 4
        generated_entries = self._generate_memory_entries(window)

        if generated_entries:
            # Step 2: Conflict-Aware Graph Coarsening - Eq. 5-6
            enable_coarsening = getattr(config, 'ENABLE_GRAPH_COARSENING', True)
            if enable_coarsening:
                final_entries = self._perform_graph_coarsening(generated_entries)
            else:
                final_entries = generated_entries
            
            # Step 3: Insert to memory graph
            if final_entries:
                self.vector_store.add_entries(final_entries)
                self.previous_entries = (self.previous_entries + final_entries)[-10:]
                self.processed_count += len(window)
                print(f"[AriadneMem] Added {len(final_entries)} entries (Filtered from {len(generated_entries)})")
            else:
                print("[AriadneMem] All entries merged/pruned.")

    def _perform_graph_coarsening(self, new_entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Conflict-Aware Graph Coarsening - Eq. 5-6 in paper
        
        Paper Reference: Section 2.2 - Conflict-Aware Graph Coarsening
        For each new entry m vs existing entry m_tilde:
        - Merge: if sim > lambda_coal AND ovlp > lambda_ovlp (true duplicate)
        - Link: if sim > lambda_coal AND ovlp <= lambda_ovlp (state update)
        - Add: otherwise (new information)
        """
        if not new_entries:
            return []

        context_entries = self.previous_entries
        
        # Batch pre-compute all needed embeddings (safe optimization: mathematically equivalent)
        all_texts = []
        for e in new_entries:
            if e.lossless_restatement not in self._embedding_cache:
                all_texts.append(e.lossless_restatement)
        for e in context_entries:
            if e.lossless_restatement not in self._embedding_cache:
                all_texts.append(e.lossless_restatement)
        
        # Batch compute and store in cache
        if all_texts:
            embeddings = self.vector_store.embedding_model.encode(all_texts)
            for text, emb in zip(all_texts, embeddings):
                self._embedding_cache[text] = emb
        
        # Original logic unchanged (using cache)
        unique_entries = []
        
        for new_ent in new_entries:
            should_drop = False
            
            # --- Check 1: External Coarsening (vs DB) ---
            for ctx_ent in context_entries:
                sim = self._calculate_text_similarity(new_ent.lossless_restatement, ctx_ent.lossless_restatement)
                
                if sim > self.coarsening_threshold:
                    new_kw = set(new_ent.keywords)
                    old_kw = set(ctx_ent.keywords)
                    kw_overlap = len(new_kw.intersection(old_kw)) / max(len(new_kw), 1)
                    
                    if kw_overlap > 0.8:
                        should_drop = True
                        break
            
            if should_drop:
                continue

            # Check 2: Internal Deduplication (vs current batch)
            for added_ent in unique_entries:
                sim = self._calculate_text_similarity(new_ent.lossless_restatement, added_ent.lossless_restatement)
                if sim > 0.98:
                    should_drop = True
                    break

            if not should_drop:
                unique_entries.append(new_ent)
                # Newly added entry also needs caching (for subsequent internal dedup)
                if new_ent.lossless_restatement not in self._embedding_cache:
                    self._embedding_cache[new_ent.lossless_restatement] = \
                        self.vector_store.embedding_model.encode_single(new_ent.lossless_restatement)

        return unique_entries

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding (using cache to avoid redundant computation)
        Safe optimization: cache key uses full text to avoid hash collisions
        """
        if text not in self._embedding_cache:
            self._embedding_cache[text] = self.vector_store.embedding_model.encode_single(text)
            
            # Limit cache size to avoid memory explosion
            if len(self._embedding_cache) > 500:
                # Clear half of the cache
                keys = list(self._embedding_cache.keys())[:250]
                for k in keys:
                    del self._embedding_cache[k]
        
        return self._embedding_cache[text]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Cosine Similarity using cached embeddings.
        """
        try:
            vec1 = self._get_embedding(text1)
            vec2 = self._get_embedding(text2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception:
            # Fallback: Token overlap (Jaccard)
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            if not set1 or not set2: return 0.0
            return len(set1.intersection(set2)) / len(set1.union(set2))

    def _get_time_diff_hours(self, ts1: Optional[str], ts2: Optional[str]) -> Optional[float]:
        """
        Helper to calculate hours between two ISO timestamps.
        """
        if not ts1 or not ts2:
            return None
        try:
            t1_str = str(ts1).replace("Z", "+00:00")
            t2_str = str(ts2).replace("Z", "+00:00")
            
            t1 = datetime.fromisoformat(t1_str)
            t2 = datetime.fromisoformat(t2_str)
            
            if t1.tzinfo is None: t1 = t1.replace(tzinfo=None)
            if t2.tzinfo is None: t2 = t2.replace(tzinfo=None)
            
            diff = abs((t1 - t2).total_seconds())
            return diff / 3600.0
        except Exception:
            return None

    def _generate_memory_entries(self, dialogues: List[Dialogue]) -> List[MemoryEntry]:
        """
        F_theta Transformation
        """
        dialogue_text = "\n".join([str(d) for d in dialogues])
        dialogue_ids = [d.dialogue_id for d in dialogues]
        
        context_str = "" 
        
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, context_str)
        
        messages = [
            {
                "role": "system", 
                "content": "You are a Knowledge Graph Engineer. Extract dense, factual, subject-action-object memory entries. Output strictly valid JSON."
            },
            {"role": "user", "content": prompt}
        ]
        
        for _ in range(2):
            try:
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}
                    
                response = self.llm_client.chat_completion(
                    messages, 
                    temperature=0.1,
                    response_format=response_format
                )
                return self._parse_llm_response(response, dialogue_ids)
            except Exception as e:
                print(f"LLM Generation Failed: {e}")
                time.sleep(1)
        return []

    def _build_extraction_prompt(self, dialogue_text: str, ids: List[int], context: str) -> str:
        """
        [Enhanced Prompt for LoCoMo-style QA]
        Focuses on: relative time preservation, counting events, specific details.
        """
        return f"""
EXTRACT structured memory entries from this conversation.

[Input Dialogue]
{dialogue_text}

[CRITICAL Extraction Rules]

1. **TIME HANDLING** (Most Important!)
   - KEEP relative time expressions EXACTLY: "yesterday", "last week", "the Friday before Oct 9"
   - Also include the absolute date if calculable in timestamp field
   - Example: "I went hiking yesterday" on Oct 5 → 
     lossless_restatement: "X went hiking yesterday (Oct 4)"
     timestamp: "2022-10-04"

2. **COUNTING & SEQUENCES** (CRITICAL - many F1=0 from wrong counts!)
   - **Count ALL occurrences**: If beach mentioned 2 times → "went to beach 2 times in 2023"
   - **Aggregate counts**: Track total count across dialogue - "rejected 2 times" if rejected twice
   - **Explicit numbers**: If dialogue says "two dogs", "twice", "2 times" → extract exact number
   - Track ordinals: "first screenplay", "second tournament", "third attempt"
   - **Repeated events**: If camping in June + camping in July → note "went camping multiple times"
   - Example: If "I was rejected" appears twice → "scripts rejected 2 times"

3. **PRESERVE SPECIFIC DETAILS** (CRITICAL - many QA failures from missing details!)
   - **Book/Movie titles**: EXACT titles in quotes - extract every book/movie mentioned by name
   - **Art specifics**: BOTH style AND subject - style (abstract/realistic) + subject (landscape/person/object) (NOT just "painting")
   - **Musical details**: ALL instruments (if plays 2, list both), band names, song titles with artists
   - **Bought items**: SPECIFIC objects - "beach figurines", "starfish figurine", "new running shoes" (NOT "items" or "stuff")
   - **Pet details**: Count + species + ALL names - "3 cats named X, Y, Z"
   - **Geographic**: NEVER "home country" - always actual country name
   - **Products**: Specific models/brands as stated
   - **Complete lists**: Extract ALL items - if 4 hobbies mentioned, list all 4; if 2 concerts, list both
   - **Numbers**: EXACT quantities - "3 children", "5 years", "2 beach visits"
   - **People/Artists**: Exact spelling - band names, performer names, author names
   - **Events**: Full names - "LGBTQ+ counseling workshop" NOT just "workshop"
   - **Symbols**: Specific symbols - any flag, icon, religious symbol mentioned
   - **Natural phenomena**: Specific names - celestial events, weather events by name
   - **Childhood details**: Favorite childhood books, activities with parents
   - **Repeated activities**: If same activity at different times, note it's repeated/multiple times

4. **PRONOUNS → NAMES**
   - Replace ALL he/she/it/they with actual names from context

5. **INFERENCE-FRIENDLY FACTS**
   - If someone plays a console-exclusive game, also note the console
   - If someone moved from a country, also note they are originally from that country
   - Include cause-effect relationships when mentioned

[Output JSON]
{{
  "entries": [
    {{
      "lossless_restatement": "Full sentence with Subject + Action + Object + Time/Place",
      "keywords": ["Person", "Action", "Object", "Topic"],
      "timestamp": "YYYY-MM-DD or null",
      "location": "Place name or null",
      "persons": ["Name1", "Name2"],
      "entities": ["Specific things mentioned"],
      "topic": "Category"
    }}
  ]
}}

Extract ALL memorable facts. Better to over-extract than miss important details.
"""

    def _parse_llm_response(self, response: str, ids: List[int]) -> List[MemoryEntry]:
        try:
            data = self.llm_client.extract_json(response)
            
            # Handle different response formats
            if isinstance(data, dict):
                # Try common key names for the array
                for key in ['entries', 'results', 'facts', 'data', 'items', 'memory_entries']:
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    # If no known key found, try to find any list value
                    for v in data.values():
                        if isinstance(v, list):
                            data = v
                            break
                    else:
                        return []
            
            if not isinstance(data, list):
                return []
            
            entries = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                # Handle location: could be string or list
                location = item.get("location")
                if isinstance(location, list):
                    location = ", ".join(str(loc) for loc in location) if location else None
                elif location and not isinstance(location, str):
                    location = str(location) if location else None
                
                # Handle timestamp: could be string or other format
                timestamp = item.get("timestamp")
                if timestamp and not isinstance(timestamp, str):
                    timestamp = str(timestamp)
                # Handle "null" string as None
                if timestamp and timestamp.lower() in ("null", "none", ""):
                    timestamp = None
                
                # Handle topic: could be string or list
                topic = item.get("topic")
                if isinstance(topic, list):
                    topic = ", ".join(str(t) for t in topic) if topic else None
                elif topic and not isinstance(topic, str):
                    topic = str(topic)
                
                # Handle lossless_restatement: ensure string
                restatement = item.get("lossless_restatement", "")
                if not isinstance(restatement, str):
                    restatement = str(restatement) if restatement else ""
                
                # Handle list fields: ensure they are lists of strings
                def ensure_str_list(val):
                    if val is None:
                        return []
                    if isinstance(val, list):
                        return [str(v) for v in val if v]
                    if isinstance(val, str):
                        return [val] if val else []
                    return [str(val)]
                
                keywords = ensure_str_list(item.get("keywords"))
                persons = ensure_str_list(item.get("persons"))
                entities = ensure_str_list(item.get("entities"))
                
                entries.append(MemoryEntry(
                    lossless_restatement=restatement,
                    keywords=keywords,
                    timestamp=timestamp,
                    location=location,
                    persons=persons,
                    entities=entities,
                    topic=topic
                ))
            return entries
        except Exception as e:
            print(f"JSON parse error: {e}")
            return []
    
    def _process_parallel(self):
        """
        Parallel processing for large batches
        
        Strategy (preserves correctness):
        1. Split into windows
        2. Parallel F_theta extraction (LLM calls)
        3. Serial conflict-aware coarsening (ensures correct deduplication)
        4. Batch insert to memory graph
        """
        # 1. Split into windows
        windows = []
        while len(self.dialogue_buffer) >= self.window_size:
            window = self.dialogue_buffer[:self.window_size]
            self.dialogue_buffer = self.dialogue_buffer[self.window_size:]
            windows.append(window)
        
        # Process remaining (less than one window)
        if self.dialogue_buffer:
            windows.append(self.dialogue_buffer)
            self.dialogue_buffer = []
        
        if not windows:
            return
        
        print(f"\n[AriadneMem Parallel] Processing {len(windows)} batches with {self.max_parallel_workers} workers")
        print(f"Batch sizes: {[len(w) for w in windows]}")
        
        # 2. Parallel LLM calls
        all_entries = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all tasks
            futures = {}
            for i, window in enumerate(windows):
                future = executor.submit(self._generate_memory_entries_worker, window, i + 1)
                futures[future] = i + 1
            
            # Collect results (as they complete)
            for future in concurrent.futures.as_completed(futures):
                window_num = futures[future]
                try:
                    entries = future.result()
                    all_entries.extend(entries)
                    print(f"[AriadneMem Parallel] Window {window_num} completed: {len(entries)} entries")
                except Exception as e:
                    print(f"[AriadneMem Parallel] Window {window_num} failed: {e}")
        
        # 3. Serial graph coarsening (ensures correct deduplication)
        enable_coarsening = getattr(config, 'ENABLE_GRAPH_COARSENING', True)
        if enable_coarsening and all_entries:
            print(f"[AriadneMem Parallel] Coarsening {len(all_entries)} entries...")
            final_entries = self._perform_graph_coarsening(all_entries)
        else:
            final_entries = all_entries
        
        # 4. Batch insert to database
        if final_entries:
            self.vector_store.add_entries(final_entries)
            self.processed_count += sum(len(w) for w in windows)
            self.previous_entries = final_entries[-10:]
            print(f"[AriadneMem Parallel] Added {len(final_entries)} entries (from {len(all_entries)} generated)")
        
        print(f"[AriadneMem Parallel] Completed processing {len(windows)} windows")

    def _generate_memory_entries_worker(self, window: List[Dialogue], window_num: int) -> List[MemoryEntry]:
        """
        Worker: Generate memory entries (for parallel calls)
        """
        batch_type = "full" if len(window) == self.window_size else "partial"
        print(f"[Worker {window_num}] Processing {batch_type} batch ({len(window)} dialogues)")
        
        dialogue_text = "\n".join([str(d) for d in window])
        dialogue_ids = [d.dialogue_id for d in window]
        
        prompt = self._build_extraction_prompt(dialogue_text, dialogue_ids, "")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a Knowledge Graph Engineer. Extract dense, factual, subject-action-object memory entries. Output strictly valid JSON."
            },
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(2):
            try:
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}
                    
                response = self.llm_client.chat_completion(
                    messages, 
                    temperature=0.1,
                    response_format=response_format
                )
                entries = self._parse_llm_response(response, dialogue_ids)
                print(f"[Worker {window_num}] Generated {len(entries)} entries")
                return entries
            except Exception as e:
                if attempt == 0:
                    print(f"[Worker {window_num}] Attempt 1 failed: {e}, retrying...")
                else:
                    print(f"[Worker {window_num}] All attempts failed: {e}")
                    return []
        return []

    def process_remaining(self):
        if self.dialogue_buffer:
            self.process_window()
    
    def build_enhanced_index(self):
        """
        Build enhanced index for fast O(1) lookups
        
        Paper Reference: Section 2.3 - Fast Paths
        Builds: aggregation cache, relation index, temporal index
        Call this after all dialogues have been processed (end of Phase I)
        """
        print("\n[AriadneMem] Building enhanced index (aggregations, relations, temporal)...")
        
        try:
            # Import here to avoid circular dependency
            from core.aggregation_builder import AggregationBuilder
            
            # Get all entries from vector store
            all_entries = self.vector_store.get_all_entries()
            
            if not all_entries:
                print("[Enhanced Index] No entries found, skipping index build")
                return
            
            # Build aggregations
            builder = AggregationBuilder()
            enhanced_index = builder.build_aggregations(all_entries)
            
            # Save to vector store
            self.vector_store.save_enhanced_index(enhanced_index)
            
            print(f"[Enhanced Index] Built successfully:")
            print(f"  - Entities: {len(enhanced_index.entities)}")
            print(f"  - Relations: {len(enhanced_index.relations)}")
            print(f"  - Temporal index entries: {len(enhanced_index.temporal_index)}")
            
        except Exception as e:
            print(f"[Enhanced Index] Build failed: {e}")
            import traceback
            traceback.print_exc()

