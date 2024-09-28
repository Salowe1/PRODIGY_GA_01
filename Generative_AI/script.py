from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = Flask(__name__)

# Load fine-tuned model
model_path = "/Users/admin/Desktop/Generative_AI/Business_model/output"
  # This is the folder where your fine-tuned model is saved
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')

    # Input validation: ensure prompt is not empty
    if not prompt.strip():
        return jsonify({'error': 'Prompt cannot be empty.'}), 400

    # Generate text using the fine-tuned model
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Set attention mask to distinguish between real tokens and padding
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,  # Pass the attention mask
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the output and return generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})


if __name__ == '__main__':
    app.run(debug=True)
