from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
DEFAULT_STORAGE_PATH = "memory_storage"

class LongTermMemory:
    id: str
    chatbot_id: str
    visit_count: int
    session_ids: List[str] # Memory units in the ltm, including the summary units.
    summary_unit_ids: List[str]
    last_session_id: Optional[str]
    creation_time: Optional[datetime]
    end_time: Optional[datetime]

    def __init__(self, chatbot_id: str, session_ids: Optional[List[str]] = None, visit_count:int = 0, creation_time: Optional[datetime] = None, end_time: Optional[datetime] = None, ltm_id: Optional[str] = None,last_session_id: Optional[str] = None,summary_unit_ids: Optional[List[str]] = None):
        self.id = ltm_id if ltm_id else str(uuid4())
        self.chatbot_id = chatbot_id
        self.summary_unit_ids = summary_unit_ids if summary_unit_ids is not None else []
        self.session_ids = session_ids if session_ids is not None else []
        self.last_session_id = last_session_id if last_session_id else None
        self.visit_count = visit_count
        self.creation_time = creation_time
        self.end_time = end_time

    def update_timestamps(self, memory_units: List['MemoryUnit']):
        valid_creation_times = [unit.creation_time for unit in memory_units if unit.creation_time]
        valid_end_times = [unit.creation_time for unit in memory_units if unit.creation_time] # Using creation time as end time for simplicity

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
            "chatbot_id": self.chatbot_id,
            "session_ids": self.session_ids,
            "summary_unit_ids": self.summary_unit_ids,
            "visit_count": self.visit_count,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "last_session_id": self.last_session_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermMemory":
        return cls(
            chatbot_id=data["chatbot_id"],
            session_ids=data.get("session_ids", []),
            visit_count=data['visit_count'],
            creation_time=datetime.fromisoformat(data["creation_time"]) if data.get("creation_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            ltm_id=data["id"],
            last_session_id=data.get("last_session_id",None),
            summary_unit_ids=data.get("summary_unit_ids",[])
        )