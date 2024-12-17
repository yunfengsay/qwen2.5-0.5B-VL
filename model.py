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