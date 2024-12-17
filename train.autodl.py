from transformers import AutoModelForCausalLM, PreTrainedModel,AutoTokenizer, PretrainedConfig, AutoProcessor, AutoModel, Trainer, TrainingArguments
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from PIL import Image 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import json
from torch.utils.data import Dataset
from typing import List, Dict, Any
from torch.utils.data.distributed import DistributedSampler

# import wandb

output_dir = '/root/autodl-tmp/save/pretrain'

# config = {
#     "learning_rate": 1e-4,
#     "epochs": 1,
#     "batch_size": 8
# }

# wandb.init(project="qwenvl",entity="shuttle",name="qwen2.5-0.5B-vl",config=config)

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, 
        llm_model_path: str = './model/Qwen2.5-0.5B-Instruct',
        vision_model_path: str = './model/siglip-so400m-patch14-384',
        freeze_vision_model: bool = True,
        image_pad_num = 729,
        **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

# 注册自定义模型的配置   
CONFIG_MAPPING.register("vlm_model", VLMConfig)

class VLM(PreTrainedModel):
    config_class = VLMConfig
    def __init__(self, config: VLMConfig):
        super().__init__(config)
        self.config = config

        # 获取当前设备
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        device = torch.device("cuda", local_rank if local_rank != -1 else 0)
    
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path, device_map={"": device})
        self.processor = AutoProcessor.from_pretrained(config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path, device_map={"": device})
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
        
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        # 冻结视觉模型
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        
        # 默认冻结所有LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False
            
        # 解冻最后几层的参数
        unfreeze_layers = 4  # 解冻最后4层
        for name, param in self.llm_model.named_parameters():
            if 'transformer.blocks' in name:  # Qwen特定的结构名称
                layer_id = int(name.split('.')[2])  # 获取层号
                if layer_id >= len(self.llm_model.transformer.blocks) - unfreeze_layers:
                    param.requires_grad = True

    def forward(self, input_ids, labels, pixel_values, attention_mask=None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b,s,d = image_embeds.shape
        
        image_embeds = image_embeds.view(b, -1, d)
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        
        # 添加梯度缩放
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

        text_embeds = text_embeds.to(image_features.dtype)
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)
    def forward(self, input_ids, labels, pixel_values, attention_mask = None):
        text_embeds = self.llm_model.get_input_embeddings()(input_ids)
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b,s,d = image_embeds.shape
        # print(image_embeds.shape)
        image_embeds = image_embeds.view(b, -1, d)
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))
        text_embeds = text_embeds.to(image_features.dtype)
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fact = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            # 修改这里：不要改变 labels 的数据类型
            loss = loss_fact(logits.view(-1, logits.size(-1)), labels.view(-1))
        # if self.training:
        #     torch.cuda.empty_cache()

        return CausalLMOutputWithPast(loss=loss, logits=logits)
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        # print(torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0]))
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # print('batch_indices shape: ', batch_indices.shape, 'image_indices shape: ', image_indices.shape) # batch_indices shape:  torch.Size([1568]) image_indices shape:  torch.Size([1568])
        # print('image_features : ', image_features.shape)  # torch.Size([8, 729, 896])
        # print('embed_dim : ', embed_dim) # 896
        # print(inputs_embeds.shape) # torch.Size([8, 249, 896])
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim) # image_features.view 后是 [5832, 896]

        return inputs_embeds
# 注册自定义模型
MODEL_MAPPING.register(VLMConfig, VLM)

class MyDataset(Dataset):
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.images_path = images_path
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversation = sample['conversations']
            q_text = self.tokenizer.apply_chat_template([{
                'role': 'system',
                'content': 'You are a helpful assistant.'
            }, {
                'role': 'user',
                'content': conversation[0]['value'],
            }], 
            tokenize=False,
            add_generating_prompt=True,
            ).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = conversation[1]['value'] + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[:-1]
            image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
            pixel_values = self.processor(text = None, images=image)['pixel_values']
        except:
            default_image = Image.new('RGB', (224, 224), color='white')
            pixel_values = self.processor(text = None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{
                'role': 'system',
                'content': 'You are a helpful assistant.'
            }, {
                'role': 'user',
                'content': '图片内容是什么\n<image>',
            }], tokenize=False, add_generating_prompt=True)
            a_text = '图片内容是' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [self.tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[:-1]

        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values,
        }

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.cat(pixel_values, dim=0),
        }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 只计算非-100部分的准确率
    valid_mask = labels != -100
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    
    accuracy = accuracy_score(valid_labels, valid_predictions)
    return {"accuracy": accuracy}

def create_train_eval_split(full_dataset, eval_ratio=0.05, seed=42):
    """
    将数据集分割成训练集和验证集
    
    Args:
        full_dataset: 完整的数据集
        eval_ratio: 验证集占比，默认5%
        seed: 随机种子，确保可复现性
    """
    dataset_size = len(full_dataset)
    eval_size = int(dataset_size * eval_ratio)
    
    # 设置随机种子
    torch.manual_seed(seed)
    
    # 生成随机索引
    indices = torch.randperm(dataset_size).tolist()
    
    # 分割数据集
    train_indices = indices[eval_size:]
    eval_indices = indices[:eval_size]
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)
    
    return train_dataset, eval_dataset
if __name__ == '__main__':
        # 添加分布式训练支持
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    
    # 初始化分布式环境
    if local_rank != -1:
        torch.distributed.init_process_group(backend="nccl")

    
    config = VLMConfig(vision_model_path='./model/siglip-so400m-patch14-384', image_pad_num=729)
    model = VLM(config).to(device)
    print(model)
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    images_path = '/root/autodl-tmp/cache/hub/datasets--liuhaotian--LLaVA-CC3M-Pretrain-595K/snapshots/814894e93db9e12a1dee78b9669e20e8606fd590/images'
    data_path = './dataset/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    
    # train_dataset = MyDataset(images_path, data_path, tokenizer, processor, config)
        # 创建完整数据集
    full_dataset = MyDataset(images_path, data_path, tokenizer, processor, config)
    
    # 分割训练集和验证集
    train_dataset, eval_dataset = create_train_eval_split(full_dataset)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(eval_dataset)}")
    
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",  # 设置为按步数评估
        eval_steps=1000,  # 每1000步评估一次
        save_strategy="steps",  # 保持相同的策略
        save_steps=1000,  # 保存步数要和评估步数相同
        save_total_limit=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=5,
        fp16=True,
        gradient_accumulation_steps=16,
        logging_steps=100,
        warmup_steps=1000,
        weight_decay=0.01,
        report_to=['tensorboard'],
        dataloader_pin_memory=True,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        logging_dir="/root/tf-logs/",
        logging_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_safetensors=True,
    )
    

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # 添加验证集
        data_collator=MyDataCollator(tokenizer),

    )
    # 检查是否存在检查点
    if os.path.exists(output_dir):
        checkpoint = None
        # 获取最新的检查点
        for checkpoint_dir in os.listdir(output_dir):
            if checkpoint_dir.startswith('checkpoint-'):
                if checkpoint is None or int(checkpoint_dir.split('-')[1]) > int(checkpoint.split('-')[1]):
                    checkpoint = checkpoint_dir
        
        if checkpoint:
            checkpoint_path = os.path.join(output_dir, checkpoint)
            print(f"Resuming from checkpoint: {checkpoint_path}")
           # 初始化模型
            config = VLMConfig(
                vision_model_path='./model/siglip-so400m-patch14-384',
                llm_model_path='./model/Qwen2.5-0.5B-Instruct',
                image_pad_num=729
            )
            model = VLM.from_pretrained(
                checkpoint_path,
                config=config,
                use_safetensors=True,
                local_files_only=True
            )

        
        
            # 创建 Trainer
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=MyDataCollator(tokenizer),
            )
            
            # 继续训练
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            # 没有检查点，从头开始训练
            trainer.train()
    else:
        # 输出目录不存在，从头开始训练
        trainer.train()
        
    # 只在主进程保存模型
    if local_rank in [-1, 0]:
      # 保存最终模型
        final_model_path = os.path.join(output_dir, 'final_model')
        model.config.save_pretrained(final_model_path)  # 保存配置
        trainer.save_model(final_model_path)
        trainer.save_state()
    # trainer.save_model('save/pretrain')
    # trainer.save_state()
