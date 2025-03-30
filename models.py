from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class ModelCard(BaseModel):
    """
    模型信息卡片类，用于描述一个可用的模型
    
    属性:
        id: 模型唯一标识符
        object: 对象类型，默认为"model"
        created: 模型创建时间戳
        owned_by: 模型所有者
        permission: 模型权限列表
        root: 根模型标识符
        parent: 父模型标识符
        description: 模型描述信息
    """
    id: str
    object: str = "model"
    created: int = Field(default=1677610602)
    owned_by: str = "owner"
    permission: List[Dict[str, Any]] = []
    root: str = None
    parent: str = None
    description: str = None

class ModelList(BaseModel):
    """
    模型列表类，用于返回可用模型的列表
    
    属性:
        object: 对象类型，默认为"list"
        data: ModelCard对象列表
    """
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    """
    聊天消息类，表示对话中的一条消息
    
    属性:
        role: 消息角色，可以是"user"、"assistant"或"system"
        content: 消息内容
        name: 可选的名称标识
        function_call: 可选的函数调用信息
    """
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class DeltaMessage(BaseModel):
    """
    增量消息类，用于流式响应中返回部分内容
    
    属性:
        role: 可选的角色信息
        content: 可选的消息内容片段
        function_call: 可选的函数调用信息
    """
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    """
    聊天完成请求类，表示API调用请求的主体
    
    属性:
        model: 使用的模型名称
        messages: 对话历史消息列表
        temperature: 温度参数，控制生成的随机性，默认0.7
        top_p: 核采样参数，默认0.9
        max_tokens: 最大生成令牌数量
        stream: 是否使用流式响应
        repetition_penalty: 重复惩罚系数，默认1.0
        tools: 可用工具列表
        n: 要生成的回复数量
    """
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
    """
    聊天完成响应选择类，表示一个完整的回复选项
    
    属性:
        index: 选择的索引
        message: 完整的回复消息
        finish_reason: 生成结束的原因，如"stop"、"length"等
    """
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponseStreamChoice(BaseModel):
    """
    聊天完成流式响应选择类，表示流式响应中的一个片段
    
    属性:
        index: 选择的索引
        delta: 增量消息内容
        finish_reason: 可选的生成结束原因
    """
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class UsageInfo(BaseModel):
    """
    用量信息类，记录令牌使用情况
    
    属性:
        prompt_tokens: 提示部分使用的令牌数
        completion_tokens: 生成部分使用的令牌数
        total_tokens: 总令牌使用数
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    """
    聊天完成响应类，表示API的完整响应
    
    属性:
        id: 响应唯一标识符
        object: 对象类型，一般为"chat.completion"
        created: 响应创建时间戳
        model: 使用的模型名称
        choices: 回复选择列表
        usage: 可选的令牌用量信息
    """
    id: str = Field(default="chatcmpl-default")
    object: str = "chat.completion"
    created: int = Field(default=1677610602)
    model: str
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    usage: Optional[UsageInfo] = None 