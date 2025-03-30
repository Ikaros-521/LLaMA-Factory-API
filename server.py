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

# 模型配置 - 可以在此处修改
# -------------------------------------
# 基础模型路径或Hugging Face模型ID
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"  
# LoRA模型路径，如不使用设为None
LORA_PATH = "saves/Qwen2-1.5B-Chat/lora/train_2024-07-09-15-38-00"  
# 推理精度，可选: "auto", "float16", "bfloat16", "float32"
INFER_DTYPE = "auto"  
# 是否使用vLLM后端进行加速
USE_VLLM = False
# 模型展示名称，用于API返回
MODEL_DISPLAY_NAME = "Qwen2-7B-Chat"
# 模型所有者信息
MODEL_OWNER = "Qwen"
# 模型描述
MODEL_DESCRIPTION = "Qwen2-7B Chat Model with LoRA"
# -------------------------------------

# 全局模型实例
model_instance = ModelInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI的生命周期管理器
    在应用启动时加载模型，应用关闭时释放模型资源
    """
    # 启动时加载模型
    logger.info("正在加载模型...")
    # 使用配置好的参数加载模型
    result = model_instance.load_model(
        base_model=BASE_MODEL, 
        adapter_path=LORA_PATH,
        infer_dtype=INFER_DTYPE,
        use_vllm=USE_VLLM
    )
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
    """
    获取可用模型列表
    
    返回:
        ModelList: 包含模型信息的列表
    """
    model_card = ModelCard(
        id=MODEL_DISPLAY_NAME,
        object="model",
        owned_by=MODEL_OWNER,
        description=MODEL_DESCRIPTION
    )
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    聊天完成接口 - 兼容OpenAI API
    
    参数:
        request: 包含消息历史和生成参数的请求对象
        
    返回:
        ChatCompletionResponse: 包含模型回复的响应对象
    
    异常:
        HTTPException: 当请求无效或处理过程中出错时抛出
    """
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
    """
    处理流式响应 - 模拟OpenAI流式API格式
    
    参数:
        request: 包含消息历史和生成参数的请求对象
        
    返回:
        AsyncGenerator: 生成流式响应的异步生成器
        
    异常:
        HTTPException: 处理过程中出错时抛出
    """
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
    