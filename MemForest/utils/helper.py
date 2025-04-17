from typing import (Dict, Iterable, List)
from MemForest.memory.memory_unit import MemoryUnit

def link_memory_units(units: Iterable[MemoryUnit]) -> List[List[MemoryUnit]]:
    """
    Links a collection of MemoryUnits into sequential chains based on pre_id and next_id.

    Handles units that might be provided out of order and identifies separate chains.

    Args:
        units: An iterable of MemoryUnit objects.

    Returns:
        A list where each element is a list representing a sequential chain of MemoryUnits.
        Units not part of any chain within the input set are returned as single-element lists.
    """
    if not units:
        return []

    unit_dict: Dict[str, MemoryUnit] = {unit.id: unit for unit in units}
    chains: List[List[MemoryUnit]] = []
    visited: set[str] = set()

    for unit_id, unit in unit_dict.items():
        if unit_id in visited:
            continue

        # Find the start of the chain this unit belongs to (within the input set)
        current = unit
        while current.pre_id and current.pre_id in unit_dict and current.pre_id not in visited:
            current = unit_dict[current.pre_id]
            # Break cycles within the input set during backward traversal
            if current == unit:
                break

                # If the found start was already visited as part of another chain, skip
        if current.id in visited:
            continue

        # Build the chain forward from the start
        chain: List[MemoryUnit] = []
        temp_current = current
        while temp_current and temp_current.id in unit_dict and temp_current.id not in visited:
            visited.add(temp_current.id)
            chain.append(temp_current)
            next_id = temp_current.next_id
            temp_current = unit_dict.get(next_id, None) if next_id else None
            # Break cycles within the input set during forward traversal
            if temp_current and temp_current.id == current.id:
                break

        if chain:
            chains.append(chain)

    # The order of chains might not be chronological
    return chains