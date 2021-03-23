'''
Finetune GPT-2 tiny, small, and medium to notes only corpus and notes + ICD appended corpus
A significant portion of this code and workflow is modeled off of the following notebook by Phil Schmid:
https://colab.research.google.com/github/philschmid/fine-tune-GPT-2/blob/master/Fine_tune_a_non_English_GPT_2_Model_with_Huggingface.ipynb#scrollTo=m9lHS0mIMak4
'''
import os
import dataset
import re
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, GPT2LMHeadModel, pipeline

model_names = ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium']
for name in model_names:
    print("Model version: " + name + "------------------------------------------")

    # Load tokenizer (use same tokenizer as Huggingface tiny GPT-2 model)
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    print("Tokenizer loaded")

    # Load model
    model = GPT2LMHeadModel.from_pretrained(name)
    print("Model loaded")

    # Set file paths
    train_path_textonly = '../Data/train_notes_textonly.txt'
    val_path_textonly = '../Data/val_notes_textonly.txt'

    train_path_withICD = '../Data/train_notes_withICD.txt'
    val_path_withICD = '../Data/val_notes_withICD.txt'

    train_path_withDIAGTEXT = '../Data/train_notes_withDIAGTEXT.txt'
    val_path_withDIAGTEXT = '../Data/val_notes_withDIAGTEXT.txt'

    # Load data into TextDatasets
    def load_dataset(train_path, val_path, tokenizer):
        train_dataset = dataset.TextDataset(
            tokenizer=tokenizer,
            file_path=train_path,
            block_size=128)

        val_dataset = dataset.TextDataset(
            tokenizer=tokenizer,
            file_path=val_path,
            block_size=128)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )
        return train_dataset, val_dataset, data_collator


    train_dataset_textonly, val_dataset_textonly, data_collator = load_dataset(train_path_textonly, val_path_textonly, tokenizer)
    train_dataset_withICD, val_dataset_withICD, data_collator = load_dataset(train_path_withICD, val_path_withICD, tokenizer)
    train_dataset_withDIAGTEXT, val_dataset_withDIAGTEXT, data_collator = load_dataset(train_path_withDIAGTEXT, val_path_withDIAGTEXT, tokenizer)
    print("TextDatasets loaded")

    # Set output directory head
    if name == 'sshleifer/tiny-gpt2':
        output_dir_head = '../Saved_Models/GPT2tiny/'
    elif name == 'gpt2':
        output_dir_head = '../Saved_Models/GPT2small/'
    elif name == 'gpt2-medium':
        output_dir_head = '../Saved_Models/GPT2medium/'

    # Train model for text only and with ICD versions of dataset
    for version in ['textonly', 'withICD', 'withDIAGTEXT']:
        if version == 'textonly':
            # Load trainer
            training_args = TrainingArguments(
                output_dir = output_dir_head + 'textonly',  # The output directory
                overwrite_output_dir = True,  # overwrite the content of the output directory
                num_train_epochs = 5,  # number of training epochs 5
                per_device_train_batch_size = 8,  # batch size for training (ORIGINAL WAS 32)
                per_device_eval_batch_size = 16,  # batch size for evaluation (ORIGINAL WAS 64)
                eval_steps = 400,  # Number of update steps between two evaluations.
                save_steps = 20000,  # after # steps model is saved
                warmup_steps = 1,  # number of warmup steps for learning rate scheduler 500
                prediction_loss_only = False
            )

            trainer = Trainer(
                model = model,
                args = training_args,
                data_collator = data_collator,
                train_dataset = train_dataset_textonly,
                eval_dataset = val_dataset_textonly
            )
        elif version == 'withICD':
            # Load trainer
            training_args = TrainingArguments(
                output_dir = output_dir_head + 'withICD',  # The output directory
                overwrite_output_dir = True,  # overwrite the content of the output directory
                num_train_epochs = 5,  # number of training epochs
                per_device_train_batch_size = 8,  # batch size for training (ORIGINAL WAS 32)
                per_device_eval_batch_size = 16,  # batch size for evaluation (ORIGINAL WAS 64)
                eval_steps = 400,  # Number of update steps between two evaluations.
                save_steps = 20000,  # after # steps model is saved
                warmup_steps = 1,  # number of warmup steps for learning rate scheduler
                prediction_loss_only = False
            )

            trainer = Trainer(
                model = model,
                args = training_args,
                data_collator = data_collator,
                train_dataset = train_dataset_withICD,
                eval_dataset = val_dataset_withICD
            )
        elif version == 'withDIAGTEXT':
            # Load trainer
            training_args = TrainingArguments(
                output_dir=output_dir_head + 'withDIAGTEXT',  # The output directory
                overwrite_output_dir=True,  # overwrite the content of the output directory
                num_train_epochs=5,  # number of training epochs
                per_device_train_batch_size=8,  # batch size for training (ORIGINAL WAS 32)
                per_device_eval_batch_size=16,  # batch size for evaluation (ORIGINAL WAS 64)
                eval_steps=400,  # Number of update steps between two evaluations.
                save_steps=20000,  # after # steps model is saved
                warmup_steps=1,  # number of warmup steps for learning rate scheduler
                prediction_loss_only=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset_withDIAGTEXT,
                eval_dataset=val_dataset_withDIAGTEXT
            )

        # Train model
        print("Training model: %s" % version + ".......")
        trainer.train()

        # Save model
        print("Saving model: %s" % version + ".......")
        trainer.save_model()


