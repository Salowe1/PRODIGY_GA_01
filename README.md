Business Plan Text Generator

This project is a web application built with Flask that leverages a fine-tuned GPT-2 model to generate business plan-related text based on a user-provided prompt. It includes a front-end interface with a responsive design and a loading animation while the model generates text.

 Features
- Text Generation: Generate business plan-related text using a fine-tuned GPT-2 model.
- Responsive Design: The front-end is designed to be responsive and adapts to different screen sizes.
- Loading Animation: A smooth loading animation is displayed while the model is generating text.
- Clear Error Messages: Users are notified when a prompt is not provided or if an error occurs during text generation.

 Technologies Used
- Python (Flask): Backend framework to serve the GPT-2 model and handle requests.
- PyTorch: Used for loading and running the GPT-2 model.
- Transformers (Hugging Face): Library to handle the GPT-2 tokenizer and model.
- HTML/CSS/JavaScript: Frontend technologies for building a dynamic user interface.
- PyPDF2: For extracting text from a PDF file to fine-tune the GPT-2 model.

 Prerequisites
- Python 3.7+
- Flask
- PyTorch
- Hugging Face Transformers
- PyPDF2

 Install Dependencies
```bash
pip install Flask torch transformers PyPDF2
```

 Project Structure
```plaintext
.
├── app.py                    # Flask app to handle text generation requests
├── static
│   └── styles.css            # Stylesheet for the web interface
├── templates
│   └── index.html            # HTML template for the UI
├── business_model
│   ├── Business_gm.pdf       # Input PDF file for training the model
│   ├── dataset.txt           # Extracted text from the PDF
│   └── output                # Folder where the fine-tuned model is saved
├── README.md                 # This README file
└── requirements.txt          # Python package requirements
```

 Steps to Fine-tune the GPT-2 Model

1. Extract Text from the PDF
   The following script extracts the text from `Business_gm.pdf` and saves it as `dataset.txt`:

    ```python
    from PyPDF2 import PdfReader
    
    pdf_path = "/path/to/Business_gm.pdf"
    output_txt = "/path/to/dataset.txt"
    
    with open(output_txt, 'w') as f:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            f.write(text)
    ```

2. Fine-tune GPT-2*
   Use the following code to fine-tune GPT-2 on the dataset extracted from the PDF:

    ```python
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    dataset_path = "/path/to/dataset.txt"
    output_dir = "/path/to/output"

    def load_dataset(file_path, tokenizer):
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)

    def create_data_collator(tokenizer):
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataset = load_dataset(dataset_path, tokenizer)
    data_collator = create_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    ```

3. Run the Flask App

    To run the web application:

    ```bash
    python script.py
    ```

    Open your browser and go to `http://127.0.0.1:5000/` to access the web interface.

 How It Works

- The user provides a business-related prompt in the input field.
- The fine-tuned GPT-2 model processes the prompt and generates a relevant response.
- The text generation process includes a loading animation until the text is fully generated.
- The generated text is displayed on the page, with error messages shown for invalid inputs.

 Usage
1. Enter a business-related prompt in the input field.
2. Click "Send" to start generating the text.
3. Wait for the loading animation to complete and view the generated text.

 Customization

You can update the styles, modify the layout, and extend functionalities by editing the files in `static/styles.css` and `templates/index.html`.

---

This should give a complete and structured guide for setting up and running your project. Let me know if you need any adjustments!
