import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from model import VLM, VLMConfig  # 从你的model.py导入

class VLMDemo:
    def __init__(self, 
                 checkpoint_path="./save/qwen2.5-0.5B-Siglip",  # 训练好的模型路径
                 llm_path="./model/Qwen2.5-0.5B-Instruct",
                 vision_path="./model/siglip-so400m-patch14-384"):
        
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置和模型
        self.config = VLMConfig(
            llm_model_path=llm_path,
            vision_model_path=vision_path
        )
        self.model = VLM(self.config)
        # 加载训练好的权重
        self.model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin"))
        self.model.to(self.device)
        self.model.eval()
        
        # 加载tokenizer和processor
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.processor = AutoProcessor.from_pretrained(vision_path)

    @torch.no_grad()
    def generate(self, image, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """生成回答"""
        try:
            # 处理图像
            if isinstance(image, str):
                image = Image.open(image)
            image = self.processor(images=image, return_tensors="pt")["pixel_values"].to(self.device)
            
            # 构建输入prompt
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # 在文本中插入图像占位符
            image_pad_token = self.tokenizer('<|image_pad|>')['input_ids'][0]
            image_pad_tokens = torch.full((1, self.config.image_pad_num), image_pad_token, device=self.device)
            input_ids = torch.cat([input_ids[:, 0:1], image_pad_tokens, input_ids[:, 1:]], dim=1)
            
            # 生成回答
            outputs = self.model.llm_model.generate(
                input_ids=input_ids,
                images=image,  # 假设模型的generate方法已适配图像输入
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取assistant部分的回答
            response = response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
            
            return response
            
        except Exception as e:
            return f"发生错误: {str(e)}"

    def create_demo(self):
        """创建Gradio界面"""
        with gr.Blocks() as demo:
            gr.Markdown("# 视觉语言模型演示")
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="上传图片")
                    
                with gr.Column():
                    text_input = gr.Textbox(
                        lines=3, 
                        label="输入提示词", 
                        placeholder="请描述你的问题..."
                    )
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.7, 
                            label="Temperature"
                        )
                        top_p = gr.Slider(
                            minimum=0.1, 
                            maximum=1.0, 
                            value=0.9, 
                            label="Top P"
                        )
                    
                    submit_btn = gr.Button("生成")
                    output_text = gr.Textbox(label="生成结果", lines=5)
            
            # 添加示例
            examples = [
                ["example_images/example1.jpg", "描述这张图片"],
                ["example_images/example2.jpg", "这张图片中有什么?"],
            ]
            gr.Examples(
                examples=examples,
                inputs=[image_input, text_input],
            )
            
            # 设置提交按钮事件
            submit_btn.click(
                fn=self.generate,
                inputs=[image_input, text_input, temperature, top_p],
                outputs=output_text
            )
            
        return demo

if __name__ == "__main__":
    # 创建演示实例
    vlm_demo = VLMDemo()
    # 启动Gradio服务
    demo = vlm_demo.create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)