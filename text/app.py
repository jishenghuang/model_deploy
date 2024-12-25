import time
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
app = FastAPI()

# 定义请求体的数据模型
class TextInput(BaseModel):
    text: str

# 加载模型和分词器
# model_name = "model_deploy/text/text_classification_model"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# 加载模型和分词器（以中文 T5 模型为例）
model_name = "google/flan-t5-xxl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto",load_in_8bit=False)

@app.post("/generate")
async def predict(input: TextInput):
    # inputs = tokenizer(input.text, return_tensors="pt").to("cuda:1")
    # 生成预测
    since = time.time()
    input_ids = tokenizer(input.text, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    time_elapsed = time.time() - since
    print(f'test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(outputs)
    # 解码生成的输出
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(prediction)
    return {"text": input.text, "generated_text": prediction}

@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return {"text": input.text, "label": prediction}
