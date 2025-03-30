from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default=1677610602)
    owned_by: str = "owner"
    permission: List[Dict[str, Any]] = []
    root: str = None
    parent: str = None
    description: str = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Any]]] = None
    n: Optional[int] = 1

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default="chatcmpl-default")
    object: str = "chat.completion"
    created: int = Field(default=1677610602)
    model: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    usage: Optional[UsageInfo] = None 