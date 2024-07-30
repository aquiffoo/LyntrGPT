import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

class LyntrGPT:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.knowledge_base = self.load_knowledge_base('data.json')

    def load_knowledge_base(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data.get('knowledge_base', '')

    def generate_response(self, prompt):
        context = self.process_prompt(prompt)
        inputs = self.tokenizer.encode_plus(context, return_tensors='pt', padding=True, truncation=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_length=100,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.85,
            temperature=0.7,
            do_sample=True,
            early_stopping=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def process_prompt(self, prompt):
        return f"{self.knowledge_base}\n{prompt}"

    def update_model(self, new_data_path):
        with open(new_data_path, 'r') as file:
            new_data = json.load(file)

        self.knowledge_base += '\n' + new_data.get('knowledge_base', '')

        with open('updated_knowledge_base.txt', 'w', encoding='utf-8') as file:
            file.write(self.knowledge_base)

        dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path='updated_knowledge_base.txt',
            block_size=128
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        training_args = TrainingArguments(
            output_dir='./results',
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset
        )
        
        trainer.train()

def main():
    lyntr = LyntrGPT()
    
    while True:
        user_input = input("You: ")
        response = lyntr.generate_response(user_input)
        print(f"LyntrGPT: {response}")

if __name__ == "__main__":
    main()
