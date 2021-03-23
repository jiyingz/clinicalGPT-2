'''
Evaluate perplexity using various models on the test corpora
Code for calculating perplexity adapted from https://huggingface.co/transformers/perplexity.html
'''
from transformers import GPT2Tokenizer, pipeline, AutoModel, GPT2LMHeadModel, DataCollatorForLanguageModeling
import torch
import numpy as np
from datetime import datetime
import time
#from tqdm import tqdm #for progress bar, if desired

device = 'cuda'

# Open output file to write results to
f = open('../Eval/perplexity.txt', 'w')
f.write("Perplexity metrics on " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")

model_names = ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium', # non-finetuned
               '../Saved_Models/GPT2tiny/', # finetuned tiny-GPT2
               '../Saved_Models/GPT2small/', # finetuned small GPT2
               '../Saved_Models/GPT2medium/'] # finetuned medium GPT2

for name in model_names:
    print("Model version: " + name + "------------------------------------------")

    for version in ['textonly', 'withICD', 'withDIAGTEXT']:
        if model_names in ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium'] and version == 'withICD':
            # Skip eval on corpus with ICD if working with non-finetuned model
            # -- since we strip away the ICDs for evaluation anyways,
            # numbers will be same as those for corpus with text only for these models
            continue

        print("Corpus: " + version)
        start = time.time()

        if name in ['sshleifer/tiny-gpt2', 'gpt2', 'gpt2-medium']:
            # Load tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(name)
            print("Tokenizer loaded")

            # Load model
            model = GPT2LMHeadModel.from_pretrained(name).to(device)
            print("Model " + name + " loaded")
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
            model = GPT2LMHeadModel.from_pretrained(name + version).to(device)
            print("Model " + name + version + " loaded")

        # Read in test dataset
        if version == 'textonly':
            test_path = '../Data/test_notes_textonly.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n')[:-1]  # list of strings (last entry seems to be empty string "")
        elif version == 'withICD':
            test_path = '../Data/test_notes_withICD.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n')[:-1]  # list of strings (last entry seems to be empty string "")
            test_data_full = [entry.split("<ICD>")[1] for entry in test_data_full] # get rid of ICDs
            #[Note: ignore the token indices sequence length too long error, since we're feeding only one window at a time later)
        elif version == 'withDIAGTEXT':
            test_path = '../Data/test_notes_withDIAGTEXT.txt'
            test_data_full = open(test_path, "r").read().split('<|endoftext|>\n')[:-1]  # list of strings (last entry seems to be empty string "")
            test_data_full = [entry.split("<ICD>")[1] for entry in test_data_full] # get rid of ICDs

        r = len(test_data_full)
        all_ppl = np.array([])
        for note_ind in range(r):
            if note_ind % 100 == 0:
                print("On note number ", note_ind + 1)

            encodings = tokenizer(test_data_full[note_ind], return_tensors='pt')

            max_length = model.config.n_positions  # max length that model can take
            stride = 512

            lls = []
            for i in range(0, encodings.input_ids.size(1), stride):  # go from index 0 to number of tokens, taking strides of size 512
                # input_ids are the windows of the actual true note, and target ids are the rest of the note with prev context masked
                # point is to get the log lik of rest of true note given the current window
                begin_loc = max(i + stride - max_length, 0)  # makes sure whatever we put into model doesn't exceed max length that model can take
                end_loc = min(i + stride, encodings.input_ids.size(1))
                trg_len = (end_loc - i) # may be different from stride on last loop
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                target_ids = input_ids.clone().to(device)  # copy of input ids but with masking context
                target_ids[:, :-trg_len] = -100  # everything before 512 character window is masked

                with torch.no_grad():
                    outputs = model(input_ids, labels=target_ids)  # returns the loss in first return arg, given the truth (labels)
                    log_likelihood = outputs[0] * trg_len

                lls.append(log_likelihood)

            ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            all_ppl = np.append(all_ppl, np.array([ppl.item()]))

        print("Average perplexity for model " + name + " on " + version + " dataset is: " + str(np.mean(all_ppl)))
        f.write("Average perplexity for model " + name + " on " + version + " dataset is: " + str(np.mean(all_ppl)) + "\n")
        end = time.time()
        print("Time elapsed: ", end - start, " seconds")

f.close()
print("DONE!")
