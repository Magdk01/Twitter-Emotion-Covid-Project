import os
import transformers
from transformers import AutoModelForMaskedLM,Trainer, TrainingArguments,AutoTokenizer
import wandb
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


if __name__ == "__main__":
    print('Import done')

    os.environ["WANDB_PROJECT"]="TwitterEmotions"
    os.environ["WANDB_LOG_MODEL"]="end"
    os.environ["WANDB_WATCH"]="false"
    os.environ['TOKENIZERS_PARALLELISM']="false"

    model_checkpoint = "cardiffnlp/twitter-roberta-base-dec2020"

    datasets = load_dataset("csv", data_files='2020_Q4_text_only_full_train.csv')
    print('Dataset loaded')

    train_size = int(len(datasets['train'])*0.10)
    test_size = int(0.1 * train_size)
    train_size -= test_size

    datasets = datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )

    print('Split made')

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    print('Tokenzier loaded')

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text",'Unnamed: 0'])

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    name_of_model = "Train_2020_Q4"

    training_args = TrainingArguments(
        f"{name_of_model}",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        report_to="wandb",
        run_name= f"{name_of_model}",
        # push_to_hub=True,
        num_train_epochs = 10,
        per_device_train_batch_size  = 12,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    trainer.train()
    wandb.finish()

    trainer.save_model()
