from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from langchain_core.messages import BaseMessage, HumanMessage, ChatMessage, SystemMessage, AIMessage
DEFAULT_STORAGE_PATH = "memory_storage"

class MemoryUnit:
    id: str
    parent_id: Optional[str]
    content: str
    creation_time: Optional[datetime]
    end_time: Optional[datetime]
    source: Optional[str] # role of message
    metadata: Dict[str, Any]
    last_visit: int # The number of rounds when the memory was last accessed
    visit_count: int
    never_delete: bool # Never delete the memory when forgetting
    children_ids: List[str]
    rank: int # 0,1,2, representing the memory of message, session or long-term memory.
    pre_id: Optional[str] # ID of the previous memory_unit in the session if exists.
    next_id: Optional[str]
    metadata: Optional[Dict[str,Any]]

    def __init__(
        self,
        content: str,
        parent_id: Optional[str] = None,
        creation_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        last_visit: int = 0,
        visit_count: int = 0,
        never_delete: bool = False,
        children_ids: Optional[List[str]] = None,
        memory_id: Optional[str] = None,
        rank: int = 0,
        pre_id: Optional[str] = None,
        next_id: Optional[str] = None,
        group_id: Optional[str] = None
    ):
        self.id = memory_id if memory_id else str(uuid4())
        self.parent_id = parent_id
        self.content = content
        self.creation_time = creation_time
        self.end_time = end_time
        self.source = source if source else "ai"
        self.metadata = metadata if metadata is not None else {}
        self.last_visit = last_visit
        self.visit_count = visit_count
        self.never_delete = never_delete
        self.children_ids = children_ids if children_ids is not None else []
        self.rank = rank
        self.pre_id = pre_id
        self.next_id = next_id
        self.group_id = group_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "content": self.content,
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "source": self.source,
            "metadata": self.metadata,
            "last_visit": self.last_visit,
            "visit_count": self.visit_count,
            "never_delete": self.never_delete,
            "children_ids": self.children_ids,
            "rank": self.rank,
            "pre_id": self.pre_id,
            "next_id": self.next_id,
            "group_id": self.group_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryUnit":
        return cls(
            content=data["content"],
            parent_id=data.get("parent_id"),
            creation_time=datetime.fromisoformat(data["creation_time"]) if data.get("creation_time") else None,
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            source=data.get("source"),
            metadata=data.get("metadata"),
            last_visit=data.get("last_visit", 0),
            visit_count=data.get("visit_count", 0),
            never_delete=data.get("never_delete", False),
            children_ids=data.get("children_ids", []),
            memory_id=data["id"],
            rank=data['rank'],
            pre_id=data["pre_id"],
            next_id=data["next_id"],
            group_id=data["group_id"]
        )

    def to_langchain_message(self) -> BaseMessage:
        metadata = {k:v for k,v in self.metadata.items()}
        metadata['source'] = self.source
        metadata['creation_time'] = self.creation_time
        metadata['end_time'] = self.end_time
        if self.source.lower() == "user":
            return HumanMessage(content=self.content, additional_kwargs=metadata)
        elif self.source.lower() == "system":
            return SystemMessage(content=self.content, additional_kwargs=metadata)
        elif self.source.lower() in ["ai","ai-summary"]:
            return AIMessage(content=self.content, additional_kwargs=metadata)
        else:
            return ChatMessage(content=self.content, role=self.source, additional_kwargs=metadata)

    @classmethod
    def from_langchain_message(cls, message: BaseMessage,
                               parent_id: Optional[str] = None,
                               creation_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None,
                               source: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None,
                               last_visit: int = 0,
                               visit_count: int = 0,
                               never_delete: bool = False,
                               children_ids: Optional[List[str]] = None,
                               memory_id: Optional[str] = None,
                               rank: int = 0,
                               pre_id: Optional[str] = None,
                               next_id: Optional[str] = None,
                               group_id: Optional[str] = None
                               ) -> "MemoryUnit":
        if not source:
            if isinstance(message, HumanMessage):
                source = "user"
            elif isinstance(message, SystemMessage):
                source = "system"
            elif isinstance(message, AIMessage):
                source = "ai"
            elif isinstance(message, ChatMessage):
                source = message.role
        return cls(
            content=message.content,
            parent_id=parent_id,
            creation_time=creation_time,
            end_time=end_time,
            source=source,
            metadata=message.additional_kwargs if not message else metadata,
            last_visit=last_visit,
            visit_count=visit_count,
            never_delete=never_delete,
            children_ids=children_ids,
            memory_id=memory_id,
            rank=rank,
            pre_id = pre_id,
            next_id = next_id,
            group_id= group_id
        )
