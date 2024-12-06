import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def verify_model():
    model_path = "./qwen_small_model"
    print(f"Verifying model in {model_path}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} not found!")
    
    # 加载模型和分词器
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 进行简单的测试推理
        print("\nPerforming test inference...")
        test_text = "你好，帮我写一首诗。"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nTest response:", response)
        
        # 打印模型信息
        model_size = sum(p.numel() for p in model.parameters()) * 2 / (1024 * 1024) # MB
        print(f"\nModel size: {model_size:.2f} MB")
        print(f"Number of parameters: {model.num_parameters():,}")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")
        return False, None, None

if __name__ == "__main__":
    success, model, tokenizer = verify_model()
    if success:
        print("\nModel verification successful!")
    else:
        print("\nModel verification failed!")