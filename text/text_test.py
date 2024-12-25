from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载 GPT2 模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 编写输入
text = "Paris is an amazing place to visit, "
encoded_input = tokenizer(text, return_tensors="pt")

# 生成输出
output = model.generate(input_ids=encoded_input["input_ids"])
decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Decoded text:", decoded_text)
