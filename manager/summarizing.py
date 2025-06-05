import datetime
from typing import List, Dict, Optional, Any, Tuple
import json
import re

try:
    from MemForest.memory.memory_unit import MemoryUnit
    from MemForest.memory.long_term_memory import LongTermMemory
    from MemForest.memory.session_memory import SessionMemory
    from MemForest.utils.llm_utils import summarize_memory, get_num_tokens
except ImportError:
    from ..memory.memory_unit import MemoryUnit
    from ..memory.long_term_memory import LongTermMemory
    from ..memory.session_memory import SessionMemory
    from ..utils.llm_utils import summarize_memory, get_num_tokens

async def summarize_memory_hierarchy(
        units_to_summarize: List[MemoryUnit],
        existing_units_map: Dict[str, MemoryUnit],  # Pass currently known/loaded units relevant to this hierarchy
        root_id: str,
        llm: 'BaseChatModel',
        max_group_size: int,
        max_token_count: int,
        history_memory: Optional[MemoryUnit] = None,
        system_message: Optional[List[Dict[str, Any]]] = None,
        role: str = "ai"
) -> Tuple[Optional[MemoryUnit], List[MemoryUnit], List[MemoryUnit]]:
    """
    Pure function to iteratively summarize memory units into a hierarchical structure.
    Does NOT perform persistence. Modifies parent_id in existing_units_map.

    Args:
        units_to_summarize: List of MemoryUnits to summarize (typically leaf nodes).
        existing_units_map: Dictionary of all units relevant to this summary context
                           (including potential parents if units_to_summarize are already summaries).
                           This dictionary will be modified (parent_id updates).
        root_id: Target ID for the final root summary unit of this hierarchy level.
        llm: Language model for summarization.
        history_memory: Optional context memory for summarization.
        system_message: Instructions for the LLM.
        role: Role of the summarizer ("ai", "user").

    Returns:
        Tuple of:
            - root_unit: The final summary unit created (or None if input was empty/invalid).
            - new_units: List of ALL summary units created during the process.
            - updated_units: List of original units whose parent_id was updated.
    """
    if not units_to_summarize:
        return None, [], []

    # Determine the rank of the summaries to be created
    base_rank = units_to_summarize[0].rank
    summary_rank = base_rank + 1

    # Sort units to process chronologically
    current_level = sorted(
        units_to_summarize,
        key=lambda x: x.creation_time if x.creation_time else datetime.datetime.min
    )
    new_summary_units = []
    updated_original_units_map = {}  # Track original units whose parent was set

    while len(current_level) > 1 or (len(current_level) == 1 and current_level[0].id != root_id):
        next_level = []
        parent_ids_processed_this_iteration = set()  # Avoid processing same parent multiple times per level

        # Group units based on size and token limits
        groups = []
        current_group = []
        current_tokens = 0
        for unit in current_level:
            if unit and unit.parent_id and unit.parent_id in parent_ids_processed_this_iteration:
                parent_unit = existing_units_map.get(unit.parent_id, None)
                if parent_unit:
                    next_level.append(parent_unit)
                    parent_ids_processed_this_iteration.add(parent_unit.id)
                    continue

            unit_tokens = get_num_tokens(unit)
            if (len(current_group) < max_group_size and
                    current_tokens + unit_tokens <= max_token_count):
                current_group.append(unit)
                current_tokens += unit_tokens
            else:
                if current_group: groups.append(current_group)
                current_group = [unit]
                current_tokens = unit_tokens
        # Add the last group
        if current_group: groups.append(current_group)

        if not groups: break

        for group in groups:
            if not group: continue

            context = []
            # Simplistic context: use history if very first summary, else use previous summary if exists
            if not next_level and not new_summary_units and history_memory:
                context = [history_memory]
            elif next_level:
                context = [next_level[-1]]

            print(f"Summarizing group of {len(group)} units (Rank {base_rank})...")
            summary_content = await summarize_memory(
                context + group,
                llm,
                system_message=system_message
            )
            print(summary_content)

            if not summary_content or not summary_content.strip():
                print("Warning: Summarization returned empty content. Skipping group.")
                continue
            # Determine time range for the summary
            valid_times = [u.creation_time for u in group if u.creation_time]
            start_time = min(valid_times) if valid_times else None
            end_time = max(valid_times) if valid_times else start_time

            summary_unit = MemoryUnit(
                content=summary_content,
                creation_time=start_time,
                end_time=end_time,
                source=f"{role}-summary",
                rank=base_rank,
                metadata={"action": "summary"},
                group_id=root_id
            )
            summary_unit.children_ids = [unit.id for unit in group]

            # Update parent-child relationships in the existing_units_map
            for unit in group:
                # Only update parent if it hasn't been set by a previous level summary
                if unit.id in existing_units_map:  # Ensure unit is in the map
                    # Check if parent is already set to avoid overwriting intermediate summaries
                    if not existing_units_map[unit.id].parent_id:
                        existing_units_map[unit.id].parent_id = summary_unit.id if len(groups) > 1 else root_id
                        updated_original_units_map[unit.id] = existing_units_map[unit.id]
                    if base_rank == 0:
                        existing_units_map[unit.id].group_id = root_id
                        updated_original_units_map[unit.id] = existing_units_map[unit.id]
                    if base_rank == 1:
                        current_group_ids = existing_units_map[unit.id].group_id
                        if current_group_ids is None:
                            new_group_ids = root_id
                        elif re.match(r"^\[\S+]$", current_group_ids):
                            try:
                                new_group_ids = json.loads(current_group_ids)
                                new_group_ids.append(root_id)
                                new_group_ids = json.dumps(new_group_ids)
                            except Exception as e:
                                print(f"Error: Current group_id for session {unit.id} is invalid. {e}")
                                new_group_ids = current_group_ids
                        else:
                            new_group_ids = current_group_ids
                        existing_units_map[unit.id].group_id = new_group_ids
                        updated_original_units_map[unit.id] = existing_units_map[unit.id]

                else:
                    print(f"Warning: Unit {unit.id} from group not found in existing_units_map.")

            next_level.append(summary_unit)
            new_summary_units.append(summary_unit)
            existing_units_map[summary_unit.id] = summary_unit

        if len(next_level) == 1:
            next_level[0].id = root_id
            next_level[0].group_id = None
        current_level = next_level  # Move to the next level of summaries

    # --- Final Root Unit Assignment ---
    final_root_unit = None
    if len(current_level) == 1:
        # The original UUID before assigning root_id
        old_id = current_level[0].id

        final_root_unit = current_level[0]
        existing_units_map.pop(final_root_unit.id, None)
        final_root_unit.id = root_id
        final_root_unit.rank = summary_rank
        existing_units_map[root_id] = final_root_unit

        if old_id != root_id:
            for child_id in final_root_unit.children_ids:
                if child_id in existing_units_map and existing_units_map[child_id].parent_id == old_id:
                    existing_units_map[child_id].parent_id = root_id
                    updated_original_units_map[child_id] = existing_units_map[child_id]

    elif len(current_level) > 1:
        print(f"Warning: Multiple units ({len(current_level)}) remaining at top level. Creating final root summary.")
        context = None
        if history_memory and not current_level:
            context = [history_memory]
        elif current_level:
            context = [current_level[-1]]
        summary_content = await summarize_memory(context + current_level, llm, system_message=system_message)
        if summary_content:
            valid_times = [u.creation_time for u in current_level if u.creation_time]
            start_time = min(valid_times) if valid_times else None
            end_time = max(valid_times) if valid_times else start_time

            final_root_unit = MemoryUnit(
                memory_id=root_id,
                content=summary_content,
                creation_time=start_time,
                end_time=end_time,
                source=f"{role}-summary",
                rank=summary_rank,
                metadata={"action": "summary"},
                group_id=root_id
            )
            final_root_unit.children_ids = [unit.id for unit in current_level]
            new_summary_units.append(final_root_unit)
            existing_units_map[root_id] = final_root_unit

            # Update parents of the current_level units
            for unit in current_level:
                if unit.id in existing_units_map:
                    # Check if parent is already set
                    if not existing_units_map[unit.id].parent_id:
                        existing_units_map[unit.id].parent_id = final_root_unit.id
                        updated_original_units_map[unit.id] = existing_units_map[unit.id]
                else:
                    print(f"Warning: Unit {unit.id} from final group not found in existing_units_map.")
        else:
            print("Error: Final root summarization failed.")

    # Collect updated original units
    updated_units_list = list(updated_original_units_map.values())

    return final_root_unit, new_summary_units, updated_units_list


# --- Session Summarization ---

# Renamed from summarize_session_memories to summarize_session
async def summarize_session(
        memory_system: 'MemorySystem',
        session_id: str,
        llm: 'BaseChatModel',
        max_group_size: int,
        max_token_count: int,
        history_memory: Optional[MemoryUnit] = None,
        system_message: Optional[List[Dict[str, Any]]] = None,
        role: str = "ai"
) -> Tuple[Optional[SessionMemory], List[MemoryUnit], List[MemoryUnit]]:
    """
    Summarizes memory units within a specific session.
    Loads data via memory_system, performs summarization, returns results.

    Args:
        memory_system: The MemorySystem instance managing data.
        session_id: The ID of the session to summarize.
        llm: Language model for summarization.
        history_memory: Optional context memory.
        system_message: Instructions for LLM.
        role: Role for summary generation.

    Returns:
        Tuple of:
         - updated_session_object: The SessionMemory object with updated timestamps and unit IDs (or None if failed).
         - new_summary_units: List of new summary units created.
         - updated_original_units: List of original units whose parent_id was updated.
    """
    # print(f"Starting summarization for session: {session_id}...")

    # 1. Load Session object
    # Assuming _get_session_memory is an async method
    session = await memory_system._get_session_memory(session_id, use_cache=True)
    if not session:
        print(f"Error: Session {session_id} not found.")
        return None, [], []

    # 2. Load MemoryUnits for this session
    session_units_map = await memory_system._load_units_for_session(session_id)
    if not session_units_map:
        print(f"Warning: No memory units found or loaded for session {session_id}. Cannot summarize.")
        # Return the session object unmodified? Or None? Let's return unmodified.
        return session, [], []

    # Include units already in cache relevant to this session
    relevant_units_map = session_units_map.copy()

    # 3. Filter units that need summarization (typically leaf nodes, rank 0)
    units_to_summarize = [
        unit for unit_id, unit in relevant_units_map.items()
        if len(unit.children_ids) == 0
    ]

    if not units_to_summarize:
        print(f"No summarizable units found for session {session_id}.")
        # Check if a session summary unit already exists
        session_summary_unit = await memory_system._get_memory_unit(session_id)
        if session_summary_unit and session_summary_unit.rank == 1:
            print(f"Session {session_id} appears to be already summarized.")
            return session, [], []  # Return existing session, no new/updated units
        else:
            print(f"No leaf units and no existing summary found for session {session_id}. Cannot summarize.")
            return session, [], []

    # print(f"Found {len(units_to_summarize)} units to summarize for session {session_id}.")

    # 4. Perform summarization using the hierarchy function
    # The session summary unit's ID should be the session_id
    root_unit, new_units, updated_units = await summarize_memory_hierarchy(
        units_to_summarize=units_to_summarize,
        existing_units_map=relevant_units_map,  # Pass the map of units involved
        root_id=session_id,  # Target ID for the session summary unit
        llm=llm,
        history_memory=history_memory,
        system_message=system_message,
        role=role,
        max_group_size=max_group_size,
        max_token_count=max_token_count
    )

    # 5. Update Session object if summarization was successful
    if root_unit:
        print(f"Session summarization successful. Root unit ID: {root_unit.id}")
        # Add IDs of newly created summary units to the session's list
        new_unit_ids = [unit.id for unit in new_units]
        session.memory_unit_ids.extend(new_unit_ids)

        # Update session timestamps based on the root summary unit
        session.creation_time = root_unit.creation_time
        session.end_time = root_unit.end_time  # Use summary end time

        # Return the updated session object and the new/updated units
        return session, new_units, updated_units
    else:
        print(f"Session summarization failed for session {session_id}.")
        # Return original session object, no new units
        return session, [], []


# --- Long-Term Memory Summarization ---

async def summarize_long_term_memory(
        memory_system: 'MemorySystem',  # Pass the memory system instance
        ltm_id: str,
        llm: 'BaseChatModel',
        max_group_size: int,
        max_token_count: int,
        history_memory: Optional[MemoryUnit] = None,
        system_message: Optional[List[Dict[str, Any]]] = None,
        role: str = "ai",
) -> Tuple[Optional[LongTermMemory], List[MemoryUnit], List[MemoryUnit]]:
    """
    Summarizes sessions within a long-term memory.
    Loads data via memory_system, calls session summarization if needed,
    summarizes session summaries, returns results.

    Args:
        memory_system: The MemorySystem instance managing data.
        ltm_id: ID of the LTM to summarize.
        llm: Language model.
        history_memory: Optional context from previous LTM summary.
        system_message: Instructions for LLM.
        role: Role for summary generation.

    Returns:
        Tuple of:
         - updated_ltm_object: The LongTermMemory object with updated timestamps/summary IDs (or None if failed).
         - new_summary_units: List of new LTM/Session summary units created.
         - updated_original_units: List of Session summary units whose parent_id was updated.
    """
    # print(f"Starting LTM summarization for: {ltm_id}...")

    # 1. Load LTM object
    ltm = memory_system.long_term_memory
    if not ltm or ltm.id != ltm_id:
        ltm = await memory_system._get_long_term_memory(ltm_id)
        if not ltm:
            print(f"Error: Long-term memory {ltm_id} not found.")
            return None, [], []

    if not ltm.session_ids:
        print(f"LTM {ltm_id} has no sessions to summarize.")
        return ltm, [], []

    # 2. Load Session objects associated with this LTM
    session_objects_map = await memory_system._load_sessions(ltm.session_ids)

    # 3. Identify/Load session summary units (MemoryUnit with id == session_id)
    session_summary_units_to_summarize: List[MemoryUnit] = []
    all_relevant_units_map: Dict[str, MemoryUnit] = {}  # To pass to hierarchy function
    sessions_needing_summary: List[str] = []
    newly_created_session_summaries: List[MemoryUnit] = []
    updated_session_units: List[MemoryUnit] = []  # Units updated during session summary

    # Sort sessions chronologically based on their metadata for ordered summarization
    sorted_session_ids = sorted(
        ltm.session_ids,
        key=lambda sid: session_objects_map.get(sid).creation_time if session_objects_map.get(
            sid) and session_objects_map.get(sid).creation_time else datetime.datetime.min
    )

    print(f"Processing {len(sorted_session_ids)} sessions for LTM summary...")
    last_session_summary_unit = history_memory  # Use overall history for first session summary

    for session_id in sorted_session_ids:
        if session_id not in session_objects_map:
            print(f"Warning: Session {session_id} listed in LTM but not loaded. Skipping.")
            continue
        session_summary_unit = await memory_system._get_memory_unit(session_id, use_cache=True)

        if session_summary_unit and session_summary_unit.rank == 1:
            # Existing session summary found
            print(f"Found existing summary for session {session_id}.")
            session_summary_units_to_summarize.append(session_summary_unit)
            all_relevant_units_map[session_id] = session_summary_unit
            last_session_summary_unit = session_summary_unit  # Use this as history for next potential summary
        else:
            # Session needs to be summarized first
            print(f"Session {session_id} needs summarization.")
            sessions_needing_summary.append(session_id)
            # Summarize the session - pass the last known summary (could be history or previous session summary)
            updated_session, new_s_units, updated_s_units = await summarize_session(
                memory_system=memory_system,
                session_id=session_id,
                llm=llm,
                history_memory=last_session_summary_unit,
                system_message=system_message,
                role=role,
                max_group_size=max_group_size,
                max_token_count=max_token_count,
            )

            # new_s_unit have intersection with updated_s_unit
            if updated_session:
                await memory_system._stage_session_memory_update(updated_session)
            if new_s_units:
                # Assuming _stage_memory_units_update is an async method
                await memory_system._stage_memory_units_update(new_s_units)
                newly_created_session_summaries.extend(new_s_units)
                # Add new units to relevant map for LTM summary
                for unit in new_s_units: all_relevant_units_map[unit.id] = unit

            if updated_s_units:
                await memory_system._stage_memory_units_update(updated_s_units, operation='edge_update')
                updated_session_units.extend(updated_s_units)
                # Update units in relevant map
                for unit in updated_s_units: all_relevant_units_map[unit.id] = unit

            # Find the actual session summary unit created (should have id == session_id)
            created_summary = next((u for u in new_s_units if u.id == session_id and u.rank == 1), None)
            if created_summary:
                session_summary_units_to_summarize.append(created_summary)
                last_session_summary_unit = created_summary  # Update history context
            else:
                print(f"Warning: Summarization of session {session_id} did not produce expected summary unit.")

    if not session_summary_units_to_summarize:
        print(f"No session summary units available to summarize for LTM {ltm_id}.")
        return ltm, newly_created_session_summaries, updated_session_units  # Return any units created during session summaries

    # print(f"Summarizing {len(session_summary_units_to_summarize)} session summaries for LTM {ltm_id}...")

    # 4. Summarize the collected session summary units
    # LTM summary unit's ID should be the ltm_id
    root_ltm_unit, new_ltm_summary_units, updated_session_summaries = await summarize_memory_hierarchy(
        units_to_summarize=session_summary_units_to_summarize,
        existing_units_map=all_relevant_units_map,  # Pass map containing session summaries
        root_id=ltm_id,  # Target ID for the LTM summary unit
        llm=llm,
        history_memory=history_memory,  # Pass original history memory here
        system_message=system_message,
        role=role,
        max_group_size=max_group_size,
        max_token_count=max_token_count
    )

    # 5. Update LTM object if summarization was successful
    if root_ltm_unit:
        # print(f"LTM summarization successful. Root unit ID: {root_ltm_unit.id}")
        # Update LTM timestamps and summary unit IDs
        ltm.creation_time = root_ltm_unit.creation_time
        ltm.end_time = root_ltm_unit.end_time
        # LTM summary unit IDs are the new summaries created at rank 2
        ltm.summary_unit_ids = [unit.id for unit in new_ltm_summary_units]

        # Combine all new units and updated units
        all_new_units = newly_created_session_summaries + new_ltm_summary_units
        all_updated_units = updated_session_units + updated_session_summaries

        return ltm, all_new_units, all_updated_units
    else:
        print(f"LTM summarization failed for {ltm_id}.")
        # Return original LTM, plus any units created/updated during session summaries
        return ltm, newly_created_session_summaries, updated_session_units