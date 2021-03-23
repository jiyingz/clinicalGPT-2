'''
Generate the rest of the note using given model, provided 100 hint tokens for each note in test dataset
In real-life applications, it is more realistic to generate, say, the next 5 tokens rather than an entire note
Adapted code for top-P/top-K generation from: https://huggingface.co/blog/how-to-generate
'''

from functools import reduce
from transformers import GPT2Tokenizer, pipeline, AutoModel, GPT2LMHeadModel, DataCollatorForLanguageModeling

model_names = ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium', # non-finetuned
               '../Saved_Models/GPT2tiny/', # finetuned tiny-GPT2
               '../Saved_Models/GPT2small/', # finetuned small GPT2
               '../Saved_Models/GPT2medium/'] # finetuned medium GPT2

for name in model_names:
    print("Model version: " + name + "------------------------------------------")

    # Save directory filename customization
    if name == 'sshleifer/tiny-gpt2':
        filename = 'tinyGPT2_nofinetune'
    elif name == 'gpt2':
        filename = 'smallGPT2_nofinetune'
    elif name == 'gpt2-medium':
        filename = 'mediumGPT2_nofinetune'
    elif name == '../Saved_Models/GPT2tiny/':
        filename = 'tinyGPT2_finetune'
    elif name == '../Saved_Models/GPT2small/':
        filename = 'smallGPT2_finetune'
    elif name == '../Saved_Models/GPT2medium/':
        filename = 'mediumGPT2_finetune'

    for version in ['textonly', 'withICD', 'withDIAGTEXT']:
        if name in ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium']:
            # Load tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(name)
            print("Tokenizer loaded")

            # Load model
            model = GPT2LMHeadModel.from_pretrained(name)
            print("Model loaded")
        else:
            # Load tokenizer (use same tokenizer as model uses)
            if name == '../Saved_Models/GPT2tiny/':
                tokenizer = GPT2Tokenizer.from_pretrained('sshleifer/tiny-gpt2')
            elif name == '../Saved_Models/GPT2small/':
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            elif name == '../Saved_Models/GPT2medium/':
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            print("Tokenizer loaded")

            # Load model
            model = GPT2LMHeadModel.from_pretrained(name + version)
            print("Model loaded")

        if version == 'textonly':
            # Read in test dataset
            test_path = '../Data/test_notes_textonly.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n') # list of strings

            # Get first 100 tokens of note as hint tokens (list of strings)
            max_model_intake = model.config.n_positions
            test_data_hints = [tokenizer.decode(tokenizer.encode(entry[:max_model_intake])[:100]) for entry in test_data_full]

            # Generate notes!
            f = open('../Generated_Text/test_generated_' + filename + '_textonly.txt', 'w')  # hack to overwrite
            f.write("")
            f.close()
            print("Cleaned the output text files...")

            print("GENERATING..............")
            f = open('../Generated_Text/test_generated_' + filename + '_textonly.txt', 'a')
            r = len(test_data_hints)

            for idx in range(r):
                if idx % 100 == 0:
                    print("Note", idx + 1)

                encoded_hint = tokenizer.encode(test_data_hints[idx], return_tensors='pt')

                # Generate from model finetuned on text only
                sample_outputs = model.generate(
                    encoded_hint,
                    do_sample = True,
                    min_length = 150,  # generate at least 50 more tokens
                    max_length = max_model_intake, # will throw an "index error: index out of range in self" if this exceeds model max intake length
                    top_p = 0.95,
                    top_k = 50,
                    num_return_sequences = 1
                )

                for i, sample_output in enumerate(sample_outputs):
                    f.write(tokenizer.decode(sample_output, skip_special_tokens=True))
                    f.write("<|endoftext|>\n")
            f.close()
            print("DONE!")
        elif version == 'withICD':
            # Read in test dataset
            test_path = '../Data/test_notes_withICD.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n')  # list of strings

            # Get first 100 tokens of note as hint tokens (list of strings)
            max_model_intake = model.config.n_positions
            test_data_hints = [tokenizer.decode(tokenizer.encode(entry.split("<ICD>")[0])) +
                               "<ICD>" +
                               tokenizer.decode(tokenizer.encode(entry.split("<ICD>")[1][:max_model_intake], max_length = 100, truncation = True))
                               for entry in test_data_full[:-1]] #modification to start counting 100 tokens from after ICD codes (last entry in test_data_full seems to be empty string "")

            # Generate notes!
            f = open('../Generated_Text/test_generated_' + filename + '_withICD.txt', 'w')  # hack to overwrite
            f.write("")
            f.close()
            print("Cleaned the output text files...")

            print("GENERATING..............")
            f = open('../Generated_Text/test_generated_' + filename + '_withICD.txt', 'a')
            r = len(test_data_hints)

            for idx in range(r):
                if idx % 100 == 0:
                    print("Note", idx + 1)

                encoded_hint = tokenizer.encode(test_data_hints[idx], return_tensors='pt')

                # Generate from model finetuned on text only
                sample_outputs = model.generate(
                    encoded_hint,
                    do_sample = True,
                    min_length = 150,  # generate at least 50 more tokens
                    max_length = max_model_intake, # will throw an "index error: index out of range in self" if this exceeds model max intake length
                    top_p = 0.95,
                    top_k = 50,
                    num_return_sequences = 1
                )

                for i, sample_output in enumerate(sample_outputs):
                    f.write(tokenizer.decode(sample_output, skip_special_tokens=True))
                    f.write("<|endoftext|>\n")
            f.close()
            print("DONE!")
        elif version == 'withDIAGTEXT':
            # Read in test dataset
            test_path = '../Data/test_notes_withDIAGTEXT.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n')  # list of strings

            # Get first 100 tokens of note as hint tokens (list of strings)
            max_model_intake = model.config.n_positions
            test_data_hints = [tokenizer.decode(tokenizer.encode(entry.split("<ICD>")[0])) +
                               "<ICD>" +
                               tokenizer.decode(tokenizer.encode(entry.split("<ICD>")[1][:max_model_intake], max_length = 100, truncation = True))
                               for entry in test_data_full[:-1]] #modification to start counting 100 tokens from after ICD codes (last entry in test_data_full seems to be empty string "")

            # Generate notes!
            f = open('../Generated_Text/test_generated_' + filename + '_withDIAGTEXT.txt', 'w')  # hack to overwrite
            f.write("")
            f.close()
            print("Cleaned the output text files...")

            print("GENERATING..............")
            f = open('../Generated_Text/test_generated_' + filename + '_withDIAGTEXT', 'a')
            r = len(test_data_hints)

            for idx in range(r):
                if idx % 100 == 0:
                    print("Note", idx + 1)

                encoded_hint = tokenizer.encode(test_data_hints[idx], return_tensors='pt')

                # Generate from model finetuned on text only
                sample_outputs = model.generate(
                    encoded_hint,
                    do_sample = True,
                    min_length = 150,  # generate at least 50 more tokens
                    max_length = max_model_intake, # will throw an "index error: index out of range in self" if this exceeds model max intake length
                    top_p = 0.95,
                    top_k = 50,
                    num_return_sequences = 1
                )

                for i, sample_output in enumerate(sample_outputs):
                    f.write(tokenizer.decode(sample_output, skip_special_tokens=True))
                    f.write("<|endoftext|>\n")
            f.close()
            print("DONE!")
