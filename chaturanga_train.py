import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model_id = "SupraLabs/Supra2-Medium-Base"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id


ds = load_dataset("mlabonne/chessllm", split="train[:200000]")
print(len(ds))
print(ds[99999])


def tokenize_function(examples):
    outputs = tokenizer(
        examples["transcript"],
        truncation=True,
        max_length=512
    ) 
    return outputs

tokenized_dataset = ds.map(
    tokenize_function,
    batched=True,
    remove_columns=ds.column_names
)

collator = DataCollatorForLanguageModeling(tokenizer, mlm=False) 


training_args = TrainingArguments(
    output_dir="./suprachess-mini",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    max_steps=5000,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    bf16=True,
    max_grad_norm=1.0,
    save_strategy="steps",
    logging_steps=500,
    save_steps=1000,
    optim="adamw_torch",
    report_to=["none"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
    processing_class=tokenizer
)

trainer.train(resume_from_checkpoint=True)
trainer.save_model("./suprachess-mini")
trainer.save_pretrained("./suprachess-mini")