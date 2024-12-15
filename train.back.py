from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PretrainedConfig, AutoProcessor, AutoModel, Trainer, TrainingArguments
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

class VLMConfig(PretrainedConfig):
    model_type = "vlm_model"
    def __init__(self, 
        llm_model_path: str = './model/Qwen2.5-0.5B-Instruct',
        vision_model_path: str = './model/siglip-so400m-patch14-384',
        freeze_vision_model: bool = True,
        image_pad_num = 49,
        **kwargs):
        self.llm_model_path = llm_model_path
        self.vision_model_path = vision_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs)

class VLM(PretrainedConfig):
    config_class = VLMConfig
    def __init__(self, config: VLMConfig , **kwargs):
        super().__init__(config = config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(config.vision_model_path)
        self.processor = AutoProcessor.from_pretrained(config.vision_model_path)
        self.llm_model = AutoModelForCausalLM.from_pretrained(config.llm_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size * 4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)

        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels, pixel_values, attention_mask = None):
        text_embeds = self.llm_model.get_input_embeddings()[input_ids]
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state
        b,s,d = image_embeds.shape
        image_embeds = image_embeds.view(b, -1, d*4)
        image_features = self.linear2(F.silu(self.linear1(image_embeds)))

        text_embeds = text_embeds.to(image_features.dtype)

        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids)
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logists = outputs[0]
        loos = None
        if labels is not None:
            loos_fact = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loos_fact(logists.view(-1, logists.size(-1)), labels.view(-1).to(logists.dtype))

        return CausalLMOutputWithPast(loss=loss, logits=logists)
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim)

        return inputs_embeds

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
            labels.append(features['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'pixel_values': torch.tensor(pixel_values, dim=0),
        }

if __name__ == '__main__':
    config = VLMConfig(vision_model_path='./model/siglip-so400m-patch14-384', image_pad_num=196)
    images_path = './dataset/LLaVA_CC3M-Pretrain-595k/images/sft_images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA_CC3M_Chinese-Pretrain-595K/Chat-translated.json'
    AutoConfig.register('vlm_model', VLMConfig)
    AutoModelForCausalLM.register(VLMConfig, VLM)
    model = AutoModelForCausalLM.from_pretrained('./save/pretrain', config=config)

    for name, param in model.named_parameters():
        if 'linear' in name or 'vision_model' in name:
            param.requires_grad = False 
        if 'llm_model' in name:
            param.requires_grad = True
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')
    print(f'可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain'
    args = TrainingArguments(
        output_dir=output_dir,
        do_train= True,
        per_device_train_batch_size=8,
        learning_rate = 1e-4,
        num_train_epochs=5,
        save_steps = 500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps = 100,
        report_to='transorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer),
    )
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()