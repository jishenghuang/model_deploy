from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 初始化对话历史
history = []
# 限制历史记录长度
def truncate_history(history, max_turns=6):
    return history[-max_turns:]

def chat_with_gpt():
    global history
    print("Chatbot: Hello! Let's chat. (Type 'exit' to stop)\n")
    
    while True:
        # 获取用户输入
        user_input = input("User: ").strip()
        
        # 检查退出条件
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        
        # 更新历史
        history.append(f"User: {user_input}")
        history = truncate_history(history)
        prompt = "\n".join(history) + "\nBot:"
        
        # 对输入进行编码
        encoded_input = tokenizer(prompt, return_tensors="pt")
        
        # 生成回复
        output = model.generate(
            **encoded_input,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample = True,
            top_p=0.9,  # 控制生成多样性
            top_k=50,
            temperature=0.7
        )
        
        # 解码回复
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 提取 Bot 的回复部分
        bot_response = response[len(prompt):].strip()
        print(f"Chatbot: {bot_response}\n")
        
        # 更新历史
        history.append(f"Bot: {bot_response}")
        history = truncate_history(history)

# 启动对话
chat_with_gpt()
