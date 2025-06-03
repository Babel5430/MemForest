from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

DEFAULT_STORAGE_PATH = "memory_storage"

class SessionMemory:
    id: str
    memory_unit_ids: List[str] # Memory units in the session, including the summary units.
    creation_time: Optional[datetime]
    end_time: Optional[datetime]

    def __init__(self, memory_unit_ids: Optional[List[str]] = None, creation_time: Optional[datetime] = None, end_time: Optional[datetime] = None, session_id: Optional[str] = None):
        self.id = session_id if session_id else str(uuid4())
        self.memory_unit_ids = memory_unit_ids if memory_unit_ids is not None else []
        self.creation_time = creation_time
        self.end_time = end_time

    def update_timestamps(self, memory_units: List['MemoryUnit']):
        valid_creation_times = [unit.creation_time for unit in memory_units if unit.creation_time] + [self.creation_time] if self.end_time else None
        valid_end_times = [unit.end_time for unit in memory_units if unit.end_time] + [self.end_time] if self.end_time else None

        if valid_creation_times:
            self.creation_time = min(valid_creation_times)
        else:
            self.creation_time = None

        if valid_end_times:
            self.end_time = max(valid_end_times)
        else:
            self.end_time = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "memory_unit_ids": self.memory_unit_ids,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMemory":
        return cls(
            memory_unit_ids=data.get("memory_unit_ids", []),
            creation_time=datetime.fromisoformat(data["creation_time"]) if data.get("creation_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            session_id=data["id"],
        )