import json
import os
from typing import Dict, Iterable, List

try:
    from MemForest.memory.memory_unit import MemoryUnit
    from MemForest.memory.long_term_memory import LongTermMemory
    from MemForest.memory.session_memory import SessionMemory
except ImportError:
    from ..memory.memory_unit import MemoryUnit
    from ..memory.long_term_memory import LongTermMemory
    from ..memory.session_memory import SessionMemory

DEFAULT_STORAGE_PATH = "memory_storage"


def _get_chatbot_path(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> str:
    return os.path.join(base_path, chatbot_id)


def _get_ltm_file_path(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> str:
    return os.path.join(_get_chatbot_path(chatbot_id, base_path), "long_term_memories.json")


def _get_sm_file_path(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> str:
    return os.path.join(_get_chatbot_path(chatbot_id, base_path), "session_memories.json")


def _get_mu_file_path(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> str:
    return os.path.join(_get_chatbot_path(chatbot_id, base_path), "memory_units.json")


def _load_json_file(file_path: str) -> Dict:
    """Loads data from a JSON file, returning {} if file missing or empty/invalid."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r', encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                return {}
            # Use json.loads directly after reading content
            return json.loads(content)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading or decoding JSON file '{file_path}': {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading JSON file '{file_path}': {e}")
        return {}


def _save_json_file(file_path: str, data: Dict):
    """Saves data to a JSON file, creating directories if needed."""
    temp_file_path = file_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(temp_file_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file_path, file_path)
    except (IOError, OSError) as e:
        print(f"Error writing JSON file '{file_path}': {e}")
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as remove_e:
                print(f"Error removing temporary file '{temp_file_path}': {remove_e}")
    except TypeError as e:
        print(f"Error serializing data for JSON file '{file_path}': {e}")


def load_long_term_memories_json(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> Dict:
    file_path = _get_ltm_file_path(chatbot_id, base_path)
    return _load_json_file(file_path)


def save_long_term_memories_json(ltms_dict: Dict, chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH):
    """Saves the entire LTM dictionary (use carefully, prefer updating single LTM)."""
    file_path = _get_ltm_file_path(chatbot_id, base_path)
    _save_json_file(file_path, ltms_dict)


def save_single_long_term_memory_json(ltm: LongTermMemory, chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH):
    """Loads the LTM file, updates one LTM object, saves back."""
    file_path = _get_ltm_file_path(chatbot_id, base_path)
    all_ltms = _load_json_file(file_path)
    all_ltms[ltm.id] = ltm.to_dict()
    _save_json_file(file_path, all_ltms)


def load_session_memories_json(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> Dict:
    file_path = _get_sm_file_path(chatbot_id, base_path)
    return _load_json_file(file_path)


def save_session_memories_json(sms_dict: Dict, chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH):
    """Saves the entire Session Memory dictionary (use carefully)."""
    file_path = _get_sm_file_path(chatbot_id, base_path)
    _save_json_file(file_path, sms_dict)


def save_session_memories_incremental_json(sessions_to_update: Iterable[SessionMemory],
                                           sessions_to_delete_ids: Iterable[str], chatbot_id: str,
                                           base_path: str = DEFAULT_STORAGE_PATH):
    """Loads the SM file, updates/deletes specified sessions, saves back."""
    file_path = _get_sm_file_path(chatbot_id, base_path)
    all_sessions = _load_json_file(file_path)
    # Update sessions
    for sm in sessions_to_update:
        all_sessions[sm.id] = sm.to_dict()
    # Delete sessions
    for sm_id in sessions_to_delete_ids:
        all_sessions.pop(sm_id, None)
    _save_json_file(file_path, all_sessions)

def load_memory_units_json(chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH) -> Dict:
    """Loads memory units from a specific session's partition file."""
    file_path = _get_mu_file_path(chatbot_id, base_path)
    return _load_json_file(file_path)


def save_memory_units_json(units_dict: Dict, chatbot_id: str, base_path: str = DEFAULT_STORAGE_PATH):
    """Saves memory units to a specific session's partition file (overwrites)."""
    file_path = _get_mu_file_path(chatbot_id, base_path)
    _save_json_file(file_path, units_dict)


def save_memory_units_incremental_json(units_to_upsert: List[MemoryUnit], units_to_delete: List[str], chatbot_id: str,
                                       base_path: str = DEFAULT_STORAGE_PATH):
    """
    Loads all memory units, updates/deletes units, saves back.
    """
    if not units_to_upsert and not units_to_delete:
        return

    units_dict = load_memory_units_json(chatbot_id, base_path)
    modified = False

    for unit_id in units_to_delete:
        if units_dict.pop(unit_id, None):
            modified = True

    for unit in units_to_upsert:
        unit_dict = unit.to_dict()
        units_dict[unit.id] = unit_dict
        modified = True

    if modified:
        save_memory_units_json(units_dict, chatbot_id, base_path)
