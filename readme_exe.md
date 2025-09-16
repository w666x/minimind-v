```多模态走起啦，vlm的预训练、微调、推理模型get。```


#### git地址


| git仓库 | 地址 | 主要功能 | star/fork数
|--|--|--|--
| minimind-v |  [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v.git) | 多模态大模型预训练微调一条龙demo | 5k/0.5k


#### 本文目录

```sh
1  minimind-v仓库介绍
  1.1  仓库简介
  1.2  demo说明
  1.3  仓库能干啥
2  常规模型训练
  2.1  模型预训练（学图像描述）
  2.2  模型微调（学看图对话方式）
  2.3  模型微调（多图对比）
3  其他
  3.1  开源数据说明
  3.2  模型说明
  3.3  faq
  3.4  环境说明
```


### minimind-v仓库介绍


#### 仓库简介


- 1. 极简VLM模型架构与训练:
    - 项目同时包含了VLM大模型的极简结构、数据集清洗、预训练 (Pretrain)、监督微调 (SFT)等全过程代码。 是一个开源VLM模型的最小实现
    - **训练VLM模型就像教小孩学看图说话，给它提供例子，慢慢引导它学习规律。**
- 2. MiniMind-V 是基于一个叫 MiniMind 的文字模型，加上一个视觉编码器（把图片变成 AI 能懂的数据），让它能同时处理文字和图片。简单来说，MiniMind-V 接收文字和图片，处理后生成回答。流程是这样的：
    - 输入：你给它文字和图片。比如，问“图片里是什么？”并附上一张图。图片在代码里用特殊符号（像“@@@…@@@” 196个）表示。
    - 处理图片：图片被送进一个叫 CLIP-ViT-Base-Patch16 的视觉编码器。它把图片切成小块（像拼图），转化成一串数字（token），AI 就能理解了。
    - 处理文字：你的问题（比如“图片里是什么？”）也被转成token。
    - 融合图文：图片token通过简单的数学运算（线性变换）调整到跟文字token“说同一种语言”。这样 AI 就能同时处理两者。
    - 输出：这些token被送进语言模型，生成回答，比如“图片里是一只狗在公园玩”。
- 3. 回复【minimind】获取git地址



#### demo说明


- 1. 模型效果试用方式如下
    - Python环境下载好 && 模型权重下载好，就可以啦
    - 下载有问题的，可以看下文 **功能拆解** 部分


```bash
git clone https://github.com/jingyaogong/minimind-v
python eval_vlm.py --load 1 --device cuda:6 # 直接加载上文下载的MiniMind2-V模型, load from transformers-hf model
# or
cd ./scripts
python web_demo_vlm.py
```

![minimindv_demo_streamlit](https://cdn.jsdelivr.net/gh/w666x/image/git/minimind_v_demo_streamlit.png)
![minimind_v_demo_cmd](https://cdn.jsdelivr.net/gh/w666x/image/git/minimind_v_demo_cmd.png)



#### 仓库能干啥


- 预训练、微调等拆解步骤，说明说句要求及训练细节


| 模块 | 功能 | 耗时/耗资源 | 数据demo | 损失函数
|---|---|---|---|---
| [无监督预训练](#模型预训练（学图像描述）) | 预训练从数据集中学习图片的通用知识 | 11G；60min/epoch/3卡 | {"conversations": [{"role": "user",  "content": "提供给定图像的简要描述。\n\<image\>"},{"role": "assistant",  "content": "橄榄油是自由使用的健康成分。"}],"image":"GCC_train_002582585.jpg"} |  CrossEntropyLoss
| [模型微调](#模型微调（学看图对话方式）) | 指令微调从真实对话数据集中学习对图片提问的真实问答格式，更符合与人类的交流习惯。 | 14G；75min/epoch/3卡 | {"conversations": [{"role": "user", "content": "context: Source Image: \<image\> Target Image: \<image\> Instruction: What ithe correct image edit instruction that can transfrom the source image to target image"},  {"role": "assistant", "content": "take the people out of the back in the photo. Remove the two people behinthe woman in the white dress and the man in the blue suit. remove people behind thcouple in the centre"}],"image": "0.jpg, 1.jpg"} |  CrossEntropyLoss



### 常规模型训练


#### 模型预训练（学图像描述）

- 1. 开始模型训练
    - **任务目标：『下一个 token 的预测』**, 预训练从595K条数据集中学习图片的通用知识，比如鹿是鹿，狗是狗。
    - 关于ddp分布式训练的，可参考文章 【分布式训练】
    - 训练时均冻结visual encoder也就是clip模型梯度，只训练Projection和LLM两部分。**预训练中，只设置Projection和LLM的最后一层参数可学习。**

```sh
cd minimind-v/trainer
python train_pretrain_vlm.py --device cuda:6 --data_path ../dataset/minimind-v_dataset/pretrain_vlm_data.jsonl --images_path ../dataset/minimind-v_dataset/pretrain_images/ --hidden_size 768 --num_hidden_layers 16 # 单节点单卡训练
export CUDA_VISIBLE_DEVICES=3,4,6
torchrun --nproc_per_node 3 --master_port=29501 train_pretrain_vlm.py  --epochs 5 --data_path ../dataset/minimind-v_dataset/pretrain_vlm_data.jsonl --images_path ../dataset/minimind-v_dataset/pretrain_images/  --hidden_size 768 --num_hidden_layers 16 --use_moe true # 单节点4卡训练

# 推理测试
python eval_vlm.py --load 0 --model_mode 0 --hidden_size 768 --num_hidden_layers 16 --device cuda:6 # 测试预训练模型效果
```


- 2. 下载模型权重文件 & 数据文件
    - 对lfs感兴趣的，可参考文章 【git-lfs】部分
    - 下载clip模型到 ./model/vision_model 目录下
    - 下载纯语言模型权重到 ./out 目录下（作为训练VLM的基座语言模型，一般可以选择有趣的项目 大模型训练part通关minimind训练得到的full_sft_768_moe.pth，直接下载的llm语言模型，加载模型的时候可以对比下模型架构和权重的key是否可以对上）
    - 下载数据集到 ./dataset 目录下

```sh
# 下载clip模型
git clone https://www.modelscope.cn/models/openai-mirror/clip-vit-base-patch16
# 下载纯语言模型
https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/blob/main/lm_768.pth

# demo模型权重下载
apt-get install git-lfs
git lfs install 
git clone https://www.modelscope.cn/gongjy/MiniMind2-V.git

# 数据下载
从下文提供的[数据集下载链接](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset)
- 下载需要的数据文件（创建`./dataset`目录）并放到`./dataset`下。
- `*.jsonl`为问答数据集，`*images`为配套的图片数据，下载完成后需要解压图像数据。
git clone https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset.git
```



- 3. 训练细节说明
    - loss: nn.CrossEntropyLoss(reduction='none')
    - optimizer: AdamW
    - hidden_size: 768 
    - num_hidden_layers: 16


| 数据量 | 模型大小 | 模型磁盘占用 | 显存占用 | 下降速度 | 训练时长/epoch | GPU数
|---|---|---|---|---|---|---
| 595375 | 0.591M | 209MB | 11G |  loss:7.793 → loss:7.306 | 60min | 3卡
| 595375 | 0.591M | 209MB | 11G |   | 300min | 1卡



- 4. 数据集demo
    - pretrain_vlm_data.jsonl,  595375条数据，数据大小129M; 每条数据对应1个图片，数据大小为3.7G
    - 数据集构造为模型输入数据集时，
    - 1）首先把 **\<image\>** 转化为 长度为196的@占位符; **任何图像都被clip模型encoder为196×768维的token**
    - 2）把conversation转化为prompot之后，构造X和Y；对于图片信息，转化为tensor维度为[3, 224, 224] 
    - 3）根据X和pixel_values经过模型推理，集合Y计算损失函数。其中 **loss_mask, 只有对话中 assistant 的回复部分（包括结束标记）会被计算损失，而 user 的输入和特殊标记不会参与损失计算**

```sh
# 原文
{
  "conversations": [
    {
      "role": "user",
      "content": "提供给定图像的简要描述。\n<image>"
    },
    {
      "role": "assistant",
      "content": "橄榄油是自由使用的健康成分。"
    }
  ],
  "image": "GCC_train_002582585.jpg"
}

# 输入到模型的文本
'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n提供给定图像的简要描述。\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@<|im_end|>\n<|im_start|>assistant\n橄榄油是自由使用的健康成分。<|im_end|>\n'

# 待训练文本
# 1）input_ids为上文的text对应的value分词编码结果
# 2）loss_mask, 只有对话中 assistant 的回复部分（包括结束标记）会被计算损失，而 user 的输入和特殊标记不会参与损失计算
# 3）eg, 即只对 <|im_start|>assistantn橄榄油是自由使用的健康成分。<|im_end|> 设置mask值为1，参与损失函数计算
loss_mask = self._generate_loss_mask(input_ids)
X = torch.tensor(input_ids[:-1], dtype=torch.long) # 
Y = torch.tensor(input_ids[1:], dtype=torch.long)
loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

# 计算损失函数
# X和Y的维度为torch.Size([16, 639])；pixel_values的维度为torch.Size([16, 1, 1, 3, 224, 224]), batch大小为16
res = model(X, pixel_values=pixel_values)
loss = loss_fct(
    res.logits.view(-1, res.logits.size(-1)),
    Y.view(-1)
).view(Y.size())
loss = (loss * loss_mask).sum() / loss_mask.sum()
```


- 5. 图片信息结合到llm的说明
    - 在minimind-v中，使用196个字符组成的 @@@...@@@ 占位符代替图像，之所以是196个字符，
    - 为啥是196呢， **任何图像都被clip模型encoder为196×768维的token**


![input](https://cdn.jsdelivr.net/gh/w666x/image/git/minimind-v-input.png)



#### 模型微调（学看图对话方式）

- 1. 开始模型训练
    - **指令微调从300K条真实对话数据集中学习对图片提问的真实问答格式，更符合与人类的交流习惯。**
    - 指令微调中，设置Projection和LLM的全部参数可学习。

```sh
cd minimind/trainer
export CUDA_VISIBLE_DEVICES=3,4,6
torchrun --nproc_per_node 3 train_sft_vlm.py --epochs 5 --hidden_size 768 --num_hidden_layers 16 --data_path ../dataset/minimind-v_dataset/sft_vlm_data.jsonl --images_path ../dataset/minimind-v_dataset/sft_images/ # 多卡
# or
python train_sft_vlm.py --epochs 4 --hidden_size 768 --num_hidden_layers 16 --device cuda:6 --data_path ../dataset/minimind-v_dataset/sft_vlm_data.jsonl --images_path ../dataset/minimind-v_dataset/sft_images/  # 单卡

# 推理测试
python eval_vlm.py --load 0 --model_mode 1 --hidden_size 768 --num_hidden_layers 16 --device cuda:6 # 测试预训练模型效果
```


- 2. 训练细节说明
    - loss: nn.CrossEntropyLoss(reduction='none')
    - optimizer: AdamW
    - hidden_size: 768 
    - num_hidden_layers: 16


| 数据量 | 模型大小 | 模型磁盘占用 | 显存占用 | 下降速度 | 训练时长/epoch | GPU数
|---|---|---|---|---|---|---
| 300k | 104M | 209M | 14G | loss:8.474 → loss:4.502  | 75min | 3卡


- 3. 数据集demo
    - sft_vlm_data.jsonl 单图指令微调数据集格式：包含300k条指令微调数据和15万张图, 问答内容经过翻译，
    - **构造mask的时候，对assistant的内容mask值为1，其他的为0**
    - **损失函数 && XY值定义同预训练一致**

```sh
# 原文
{
  "conversations": [
    {
      "role": "user",
      "content": "闹钟的位置对睡眠质量有什么影响？<image>"
    },
    {
      "role": "assistant",
      "content": "把数字闹钟放在床头柜或光线充足的木桌上，会在几个方面影响睡眠质量。红色发光屏幕上显示着时间（晚上10点），发出的光可能让人分心或者引起不适，使人难以入睡。夜间暴露于哪怕少量的光线中都能干扰褪黑素的产生，而褪黑素有调节睡眠周期的作用。为了确保更好的睡眠质量，可以将时钟屏幕调暗、把它远离床摆放，或是移到房间里视线较不那么好的地方。此外，睡觉前让房间保持明亮也会扰乱自然的睡眠循环，所以关掉或调低任何不必要的灯光能够改善整体的睡眠环境。"
    }
  ],
  "image": "train-00000-of-00001_image_0_0.jpg"
}

# 输入到模型的文本, prompt
'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n闹钟的位置对睡眠质量有什么影响？@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@<|im_end|>\n<|im_start|>assistant\n把数字闹钟放在床头柜或光线充足的木桌上，会在几个方面影响睡眠质量。红色发光屏幕上显示着时间（晚上10点），发出的光可能让人分心或者引起不适，使人难以入睡。夜间暴露于哪怕少量的光线中都能干扰褪黑素的产生，而褪黑素有调节睡眠周期的作用。为了确保更好的睡眠质量，可以将时钟屏幕调暗、把它远离床摆放，或是移到房间里视线较不那么好的地方。此外，睡觉前让房间保持明亮也会扰乱自然的睡眠循环，所以关掉或调低任何不必要的灯光能够改善整体的睡眠环境。<|im_end|>\n'

# 待训练文本
# 1) input_ids为上文的text对应的value分词编码结果
# 2) loss_mask, 只有对话中 assistant 的回复部分（包括结束标记）会被计算损失，而 user 的输入和特殊标记不会参与损失计算
# 3) eg, 即只对 <|im_start|>assistantn橄榄油是自由使用的健康成分。<|im_end|> 设置mask值为1，参与损失函数计算
loss_mask = self._generate_loss_mask(input_ids)
X = torch.tensor(input_ids[:-1], dtype=torch.long) # 
Y = torch.tensor(input_ids[1:], dtype=torch.long)
loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

# 计算损失函数
# X和Y的维度为torch.Size([16, 639])；pixel_values的维度为torch.Size([16, 1, 1, 3, 224, 224]), batch大小为16
res = model(X, pixel_values=pixel_values)
loss = loss_fct(
    res.logits.view(-1, res.logits.size(-1)),
    Y.view(-1)
).view(Y.size())
loss = (loss * loss_mask).sum() / loss_mask.sum()
```


#### 模型微调（多图对比）

- 1. 开始模型训练
    - 多图微调提供demo：鸟类对比数据集，长度为13.6k的真实问答格式。
    - 多图数据集规模相对较小且为英文对话，**数据集仅包含两图对比的场景**，因此微调效果有限，这里只提供一种参考思路。
    - 指令微调中，设置Projection和LLM的全部参数可学习。

```sh
cd minimind/trainer
export CUDA_VISIBLE_DEVICES=3,4,6
torchrun --nproc_per_node 3 train_sft_vlm.py --epochs 4 --hidden_size 768 --num_hidden_layers 16 --data_path ../dataset/minimind-v_dataset/sft_vlm_data_multi.jsonl --images_path ../dataset/minimind-v_dataset/sft_multi_images/ # 多卡
# or
python train_sft_vlm.py --epochs 4 --hidden_size 768 --num_hidden_layers 16 --device cuda:6 --data_path ../dataset/minimind-v_dataset/sft_vlm_data_multi.jsonl --images_path ../dataset/minimind-v_dataset/sft_multi_images/  # 单卡

# 推理测试
python eval_vlm.py --load 0 --model_mode 1 --use_multi 2 --hidden_size 768 --num_hidden_layers 16 --device cuda:6 # 测试预训练模型效果
```

- 2. 训练细节说明
    - loss: nn.CrossEntropyLoss(reduction='none')
    - optimizer: AdamW
    - hidden_size: 768 
    - num_hidden_layers: 16


| 数据量 | 模型大小 | 模型磁盘占用 | 显存占用 | 下降速度 | 训练时长/epoch | GPU数
|---|---|---|---|---|---|---
| 3K | 104M | 209MB | 16G | loss:8.671 → loss:5.183 | 3min | 3卡



- 3. 数据集demo
    - sft_vlm_data_multi.jsonl 多图指令微调数据集格式：13.6k的真实问答格式
    - **构造mask的时候，对assistant的内容mask值为1，其他的为0**
    - **损失函数 && XY值定义同预训练一致**

```sh
# 原文
{
  "conversations": [
    {
      "role": "user",
      "content": "context: Source Image: <image> Target Image: <image> Instruction: What is the correct image edit instruction that can transfrom the source image to target image?"
    },
    {
      "role": "assistant",
      "content": "take the people out of the back in the photo. Remove the two people behind the woman in the white dress and the man in the blue suit. remove people behind the couple in the centre"
    }
  ],
  "image": "0.jpg, 1.jpg"
}

# 输入到模型的文本, prompt
'<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\ncontext: Source Image: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Target Image: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Instruction: What is the correct image edit instruction that can transfrom the source image to target image?<|im_end|>\n<|im_start|>assistant\ntake the people out of the back in the photo. Remove the two people behind the woman in the white dress and the man in the blue suit. remove people behind the couple in the centre<|im_end|>\n'
```



### 其他


#### 开源数据说明


- 1. 开源数据集
    - 基于[Chinese-LLaVA-Vision](https://huggingface.co/datasets/LinkSoul/Chinese-LLaVA-Vision-Instructions) 下载的
    - 包含约57万张预训练图像，来自CC-3M和COCO 2014；



- 2. 开源数据集
    - 基于 [llava-en-zh-300k](https://huggingface.co/datasets/BUAADreamer/llava-en-zh-300k) 下载的
    - 包含300k条指令微调数据和15万张图像。问答内容经过翻译，对中文支持更友好，进一步经过整理并`resize`。



#### 模型说明


- 1. 模型框架
  - **MiniMind-V的结构仅增加Visual Encoder和特征投影两个子模块，增加模态混合分支，以支持多种模态信息的输入**
  - 如下所示，为dense和moe模型结构



![LLM-structure](https://cdn.jsdelivr.net/gh/w666x/image/git/VLM-structure.png)
![LLM-structure](https://cdn.jsdelivr.net/gh/w666x/image/git/VLM-structure-moe.png)



- 2. 文本图片对齐方式
  - 本文，具体使用[clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)，一种基于 ViT-B/16 架构的经典Visual Encoder用于描述图像文本信息。
  - 本文输入的图像尺寸为224x224，因为划分的Patch是16×16，所以会产生14*14=196个token作为encoder编码层的输入，最终产生1×768维的嵌入向量用于和文本对计算误差。
  - **我们并不需要最终嵌入表示，因此只取encoder层的输出，也就是VIT核心主干的输出特征即可**。
  - 它拿到前一层维度196×768大小的特征，我们把它作为196个visual token输入MiniMind-V。与LLM的结合在获取图像encoder特征后，
      - 一方面需要把768维度的visual token对齐到LLM的文本token，
      - 另一方面，要将图像特征映射到与文本embedding相同的空间，即文本token和原生的视觉token需要磨合并不能直接地一视同仁，可以称之为跨模态的特征对齐。
  - [LlaVA-1](https://arxiv.org/pdf/2304.08485)使用简单的无偏线性变换完成了这一操作，效果很不错，MiniMind-V同样如此。



![llava-structure](https://cdn.jsdelivr.net/gh/w666x/image/git/llava-structure.png)



- 3. text & vision模型结构


```sh
CLIPModel(
  (text_model): CLIPTextTransformer(
    (embeddings): CLIPTextEmbeddings(
      (token_embedding): Embedding(49408, 512)
      (position_embedding): Embedding(77, 512)
    )
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPSdpaAttention(
            (k_proj): Linear(in_features=512, out_features=512, bias=True)
            (v_proj): Linear(in_features=512, out_features=512, bias=True)
            (q_proj): Linear(in_features=512, out_features=512, bias=True)
            (out_proj): Linear(in_features=512, out_features=512, bias=True)
          )
          (layer_norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
          )
          (layer_norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  )
  (vision_model): CLIPVisionTransformer(
    (embeddings): CLIPVisionEmbeddings(
      (patch_embedding): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16), bias=False)
      (position_embedding): Embedding(197, 768)
    )
    (pre_layrnorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (encoder): CLIPEncoder(
      (layers): ModuleList(
        (0-11): 12 x CLIPEncoderLayer(
          (self_attn): CLIPSdpaAttention(
            (k_proj): Linear(in_features=768, out_features=768, bias=True)
            (v_proj): Linear(in_features=768, out_features=768, bias=True)
            (q_proj): Linear(in_features=768, out_features=768, bias=True)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
          (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (mlp): CLIPMLP(
            (activation_fn): QuickGELUActivation()
            (fc1): Linear(in_features=768, out_features=3072, bias=True)
            (fc2): Linear(in_features=3072, out_features=768, bias=True)
          )
          (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (post_layernorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (visual_projection): Linear(in_features=768, out_features=512, bias=False)
  (text_projection): Linear(in_features=512, out_features=512, bias=False)
)
```



#### faq

- 1. 报缓存不够的话，

```sh
问题点：RuntimeError: DataLoader worker (pid 29034) is killed by signal: Bus error. It is possible that dataloaders workers are out of shared memory. Please try to raise your shared memory limit
解决方案：
# 1. 获取容器PID
pid=$(docker inspect -f '{{.State.Pid}}' docker_test)

# 2. 在容器命名空间内创建新共享内存
sudo nsenter -m -t $pid -- sh -c "
   # 卸载现有/dev/shm（如果存在）
   umount /dev/shm 2>/dev/null || true;
   
   # 创建新的tmpfs共享内存
   mount -t tmpfs -o size=2g tmpfs /dev/shm;
   
   # 设置正确权限
   chmod 1777 /dev/shm
"

# 3. 验证
docker exec docker_test df -h /dev/shm
```


#### 环境说明



- 1. 环境说明

```sh
ubuntu1~11.4.0
NVIDIA: A100
docker: 27.5.1
Python: 3.12.11
CUDA Version: 12.2
nvcc: 12.8
imgae: nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
```

- 2. python环境
    - python库版本，请看requirements.txt


```sh
torch==2.2.2
openai==1.59.6
transformers==4.48.0
```
