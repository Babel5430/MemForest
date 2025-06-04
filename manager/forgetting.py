"""
Module for identifying less important memories to forget based on LTM scope.
Relies on MemorySystem for data loading and context.
"""

from typing import List, Tuple, Set
from datetime import datetime

try:
    from MemForest.memory.memory_unit import MemoryUnit
    from MemForest.memory.long_term_memory import LongTermMemory
    from MemForest.memory.session_memory import SessionMemory
except ImportError:
    from ..memory.memory_unit import MemoryUnit
    from ..memory.long_term_memory import LongTermMemory
    from ..memory.session_memory import SessionMemory

# Constants remain the same
MIN_DELETE_PERCENTAGE = 0.0
MAX_DELETE_PERCENTAGE = 1.0


async def forget_memories(
        memory_system: 'MemorySystem',
        ltm_id: str,
        delete_percentage: float = 0.15,
) -> Tuple[List[str], Set[str], Set[str]]:
    """
    Identifies less important memories within the scope of a specific LTM to forget.
    Loads data via memory_system, performs selection logic.

    Args:
        memory_system: The MemorySystem instance managing data.
        ltm_id: ID of the long-term memory instance to clean.
        delete_percentage: Percentage of eligible leaf nodes to target for removal (0-1).

    Returns:
        Tuple of:
         - deleted_unit_ids (List[str]): IDs of the units selected for deletion.
         - updated_parent_ids (Set[str]): IDs of parent units needing child list update.
         - updated_session_ids (Set[str]): IDs of sessions needing unit list update.
    """
    # print(f"Starting forgetting process for LTM: {ltm_id}...")

    # Validate input
    if not MIN_DELETE_PERCENTAGE <= delete_percentage <= MAX_DELETE_PERCENTAGE:
        raise ValueError(f"Delete percentage must be between {MIN_DELETE_PERCENTAGE} and {MAX_DELETE_PERCENTAGE}")
    if delete_percentage == 0.0:
        print("Delete percentage is 0. No units will be forgotten.")
        return [], set(), set()

    # 1. Load LTM object
    ltm = memory_system.long_term_memory
    if not ltm or ltm.id != ltm_id:
        # Assuming _get_long_term_memory is an async method
        ltm = await memory_system._get_long_term_memory(ltm_id)
        if not ltm:
            print(f"Error: LTM {ltm_id} not found for forgetting.")
            return [], set(), set()

    if not ltm.session_ids:
        print(f"LTM {ltm_id} has no sessions. Nothing to forget.")
        return [], set(), set()

    session_ids_set_in_ltm = set(ltm.session_ids)
    # print(f"Loaded {len(session_objects_map)} sessions for LTM {ltm_id}.")

    # 2. Load ALL MemoryUnits associated with these sessions

    all_units_map = memory_system.memory_units_cache
    # print(f"Loaded {len(all_units_map)} memory unit objects.")

    # 3. Identify eligible leaf nodes within this LTM's scope
    eligible_leaves: List[MemoryUnit] = []
    for unit_id, unit in all_units_map.items():
        # Check if it's a leaf node (no children) AND not marked 'never_delete'
        if (not unit.children_ids and
                not unit.never_delete and unit.rank == 0):
            eligible_leaves.append(unit)

    # print(f"Found {len(eligible_leaves)} eligible leaf units for forgetting.")
    if not eligible_leaves:
        return [], set(), set()

    # 4. Sort eligible units by importance (least important first)
    # Importance: last_visit (ascending), then visit_count (ascending), then creation_time (ascending)
    sorted_leaves = sorted(
        eligible_leaves,
        key=lambda x: (
            x.last_visit,
            x.visit_count,
            x.creation_time if x.creation_time else datetime.min  # Handle None times
        )
    )

    # 5. Calculate how many to delete
    delete_count = max(1, int(len(sorted_leaves) * delete_percentage)) if delete_percentage > 0 else 0
    delete_count = min(delete_count, len(sorted_leaves))

    if delete_count == 0:
        print("Calculated delete count is 0.")
        return [], set(), set()

    # print(f"Targeting {delete_count} units for deletion ({delete_percentage*100:.1f}% of eligible leaves).")

    # 6. Select units and determine side effects
    units_to_delete = sorted_leaves[:delete_count]
    deleted_unit_ids: List[str] = [unit.id for unit in units_to_delete]
    updated_parent_ids: Set[str] = set()
    updated_group_ids: Set[str] = set()

    for unit in units_to_delete:
        # Find session and mark for update
        group_id = unit.group_id
        if not group_id or group_id not in session_ids_set_in_ltm:
            print(f"Error: Could not find session mapping for eligible unit {unit.id}.")
            continue
        else:
            updated_group_ids.add(group_id)

        if unit.parent_id and unit.parent_id in all_units_map:  # Check parent is loaded
            updated_parent_ids.add(unit.parent_id)
        elif unit.parent_id:
            print(
                f"Warning: Parent {unit.parent_id} of unit {unit.id} not found in loaded units. Cannot mark for update.")


    # print(f"Selected {len(deleted_unit_ids)} units for deletion.")
    # print(f"Identified {len(updated_parent_ids)} parent units needing update.")
    # print(f"Identified {len(updated_session_ids)} sessions needing update.")

    # 7. Return the lists of IDs for MemorySystem to handle persistence
    return deleted_unit_ids, updated_parent_ids, updated_group_ids