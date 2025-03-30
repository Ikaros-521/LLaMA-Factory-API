from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class GenerationConfig:
    """
    生成配置类，用于控制文本生成的参数
    
    参数:
        max_length: 生成文本的最大长度
        top_p: 用于核采样的概率阈值
        temperature: 生成多样性控制，较高的值会增加随机性
        repetition_penalty: 重复惩罚因子，大于1时会降低重复单词的可能性
    """
    max_length: int = 32768
    top_p: float = 0.9
    temperature: float = 0.7
    repetition_penalty: float = 1.0

def format_messages(messages: List[Dict[str, str]]) -> str:
    """
    将消息列表格式化为Qwen2模型需要的格式
    
    参数:
        messages: 消息列表，每个消息包含role和content字段
        
    返回:
        格式化后的文本字符串
    """
    formatted_text = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text

class ModelInference:
    """
    模型推理类，负责加载模型、管理和执行推理
    """
    def __init__(self):
        """初始化推理类，设置基本属性"""
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.generation_config = GenerationConfig()
        
    def load_model(
        self,
        base_model: str = "Qwen/Qwen2-1.5B-Chat",
        adapter_path: str = "saves/Qwen2-1.5B-Chat/lora/train_2024-07-09-15-38-00",
        infer_dtype: str = "auto",
        use_vllm: bool = False
    ) -> str:
        """
        加载语言模型和分词器
        
        参数:
            base_model: 基础模型路径或名称
            adapter_path: LoRA适配器路径，如果不使用可设为None
            infer_dtype: 推理数据类型，可以是'auto'、'float16'、'bfloat16'或'float32'
            use_vllm: 是否使用vLLM后端来加速推理
            
        返回:
            加载状态信息字符串
        """
        try:
            # 检查并安装必要的依赖
            if not use_vllm:  # 使用HuggingFace后端
                
                # 设置计算数据类型
                if infer_dtype == "auto":
                    compute_dtype = (torch.bfloat16 
                                   if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                                   else torch.float16)
                else:
                    compute_dtype = (torch.float16 if infer_dtype == "float16" else 
                                   torch.bfloat16 if infer_dtype == "bfloat16" else 
                                   torch.float32)
                
                # 配置量化参数
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model,
                    trust_remote_code=True,
                    use_fast=False
                )
                
                # 加载基础模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=quantization_config, # 不量化就注释
                    torch_dtype=compute_dtype,
                    low_cpu_mem_usage=True
                )
                
                if adapter_path:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        adapter_path,
                        torch_dtype=compute_dtype
                    )
                
                # 设置模型评估模式
                self.model.eval()
            
            else:  # 使用vllm后端
                try:
                    from vllm import LLM, SamplingParams
                    self.model = LLM(
                        model=base_model,
                        trust_remote_code=True,
                        tensor_parallel_size=1,
                        dtype=infer_dtype
                    )
                    self.tokenizer = self.model.get_tokenizer()
                except ImportError:
                    return "使用vLLM后端需要先安装vllm包"
            
            self.loaded = True
            return "模型加载成功！"
        except Exception as e:
            return f"模型加载失败：{str(e)}"
    
    def unload_model(self) -> str:
        """
        卸载模型并释放GPU内存
        
        返回:
            卸载状态信息字符串
        """
        if not self.loaded:
            return "模型尚未加载"
        
        try:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            self.loaded = False
            return "模型已卸载"
        except Exception as e:
            return f"模型卸载失败：{str(e)}"
    
    def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> str:
        """
        执行对话推理
        
        参数:
            query: 用户输入的查询文本
            history: 对话历史，格式为包含role和content字段的字典列表
            stream: 是否使用流式输出（当前实现中未使用）
            
        返回:
            模型生成的回复文本
        """
        if not self.loaded:
            return "请先加载模型"
        
        if history is None:
            history = []
        
        messages = history + [{"role": "user", "content": query}]
        prompt = format_messages(messages)
        
        try:
            with torch.inference_mode():
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_length,
                    do_sample=True,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    repetition_penalty=self.generation_config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                # 移除可能的助手标记
                response = response.split("<|im_end|>")[0].strip()
            return response
        except Exception as e:
            return f"生成回复时发生错误：{str(e)}"

def main():
    """
    主函数，用于命令行交互式推理
    在这里可以方便地修改模型和LoRA权重路径
    """
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
    # -------------------------------------
    
    inference = ModelInference()
    
    logger.info("正在加载模型...")
    # 使用上方定义的模型参数加载模型
    result = inference.load_model(
        base_model=BASE_MODEL,
        adapter_path=LORA_PATH,
        infer_dtype=INFER_DTYPE,
        use_vllm=USE_VLLM
    )
    logger.info(result)
    
    if inference.loaded:
        history = []
        while True:
            query = input("\n请输入您的问题（输入'quit'退出，输入'reload'重新加载模型，输入'clear'清除历史）: ")
            
            if query.lower() == 'quit':
                inference.unload_model()
                break
            elif query.lower() == 'reload':
                logger.info(inference.unload_model())
                logger.info("重新加载模型...")
                logger.info(inference.load_model(
                    base_model=BASE_MODEL,
                    adapter_path=LORA_PATH,
                    infer_dtype=INFER_DTYPE,
                    use_vllm=USE_VLLM
                ))
                continue
            elif query.lower() == 'clear':
                history = []
                logger.info("对话历史已清除")
                continue
                
            response = inference.chat(query, history)
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": response})
            logger.info("\n回答:", response)

if __name__ == "__main__":
    main() 