import time
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, AsyncGenerator
from sse_starlette.sse import EventSourceResponse

from models import (
    ModelCard, 
    ModelList, 
    ChatCompletionRequest, 
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    DeltaMessage,
    UsageInfo
)

from inference import ModelInference

from loguru import logger

# 全局模型实例
model_instance = ModelInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    logger.info("正在加载模型...")
    # 加载模型 base_model是模型名称，adapter_path是lora路径
    result = model_instance.load_model(base_model="Qwen/Qwen2-1.5B-Instruct", adapter_path="saves/Qwen2-1.5B-Chat/lora/train_2024-07-09-15-38-00")
    logger.info(result)
    yield
    # 关闭时释放模型
    model_instance.unload_model()

app = FastAPI(lifespan=lifespan)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """获取可用模型列表"""
    model_card = ModelCard(
        id="Qwen2-7B-Chat",
        object="model",
        owned_by="Qwen",
        description="Qwen2-7B Chat Model with LoRA"
    )
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """聊天完成接口"""
    if len(request.messages) < 1:
        raise HTTPException(status_code=400, detail="消息列表不能为空")
    
    try:
        # 处理流式请求
        if request.stream:
            return EventSourceResponse(
                stream_chat_response(request),
                media_type="text/event-stream"
            )
        
        # 处理普通请求
        response = model_instance.chat(
            query=request.messages[-1].content,
            history=[{"role": m.role, "content": m.content} for m in request.messages[:-1]]
        )
        
        message = ChatMessage(
            role="assistant",
            content=response
        )
        
        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        
        # 计算token数量（这里简单估算）
        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(response.split())
        
        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        return ChatCompletionResponse(
            model=request.model,
            choices=[choice],
            usage=usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_chat_response(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """处理流式响应"""
    try:
        # 发送开始标记
        chunk = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            model=request.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(role="assistant"),
                    finish_reason=None
                )
            ],
            object="chat.completion.chunk"
        )
        yield json.dumps(chunk.model_dump(exclude_unset=True))
        
        # 获取响应
        response = model_instance.chat(
            query=request.messages[-1].content,
            history=[{"role": m.role, "content": m.content} for m in request.messages[:-1]],
            stream=True
        )
        
        # 按字符输出
        for char in response:
            if not char:
                continue
                
            chunk = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                model=request.model,
                choices=[
                    ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=char),
                        finish_reason=None
                    )
                ],
                object="chat.completion.chunk"
            )
            yield json.dumps(chunk.model_dump(exclude_unset=True))
            
        # 发送结束标记
        chunk = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            model=request.model,
            choices=[
                ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason="stop"
                )
            ],
            object="chat.completion.chunk"
        )
        yield json.dumps(chunk.model_dump(exclude_unset=True))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("正在启动服务器...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
    