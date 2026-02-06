from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime
import uuid


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionData(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    messages: list[Message] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        self.last_activity = datetime.utcnow()

    def get_context_messages(self, max_messages: int = 10) -> list[dict]:
        """Get recent messages for context"""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [{"role": m.role, "content": m.content} for m in recent]


class SessionCreate(BaseModel):
    metadata: dict = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    metadata: dict
