from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load the pre-trained model
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define chatbot response function
def chatbot_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Gradio interface
def interface_function(user_input):
    return chatbot_response(user_input)

gr.Interface(
    fn=interface_function,
    inputs="text",
    outputs="text",
    title="Simple Cloud Chatbot",
    description="Ask me anything about cloud computing!",
).launch()
