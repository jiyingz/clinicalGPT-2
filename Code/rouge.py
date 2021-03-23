'''
Calculate ROUGE-1 and ROUGE-2 scores
Code adapted from https://pypi.org/project/rouge-score/
'''
from rouge_score import rouge_scorer
from datetime import datetime
import numpy as np
import time

# Open output file to write results to
f = open('../Eval/rouge.txt', 'w')
f.write("ROUGE metrics on " + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "\n")

# Define a scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

for corpus in ['textonly', 'withICD', 'withDIAGTEXT']:
    # Get datasets
    if corpus == 'textonly':
        f.write("Calculating ROUGE for text only test set ---------------------------\n")

        # Load data
        test_path_groundtruth = '../Data/test_notes_textonly.txt'
        test_data_groundtruth = open(test_path_groundtruth, "r").read().split('<|endoftext|>\n') # list of strings

        # Generated texts to compare to
        corpi = ['../Generated_Text/test_generated_tinyGPT2_nofinetune_textonly.txt',
                 '../Generated_Text/test_generated_smallGPT2_nofinetune_textonly.txt',
                 '../Generated_Text/test_generated_mediumGPT2_nofinetune_textonly.txt',
                 '../Generated_Text/test_generated_tinyGPT2_finetune_textonly.txt',
                 '../Generated_Text/test_generated_smallGPT2_finetune_textonly.txt',
                 '../Generated_Text/test_generated_mediumGPT2_finetune_textonly.txt']
    elif corpus == 'withICD':
        f.write("Calculating ROUGE for ICD appended test set ---------------------------\n")

        # Load data
        test_path_groundtruth = '../Data/test_notes_withICD.txt'
        test_data_groundtruth = open(test_path_groundtruth, "r").read().split('<|endoftext|>\n') # list of strings
        test_data_groundtruth = [entry.split("<ICD>")[1] for entry in test_data_groundtruth[:-1]]  # get rid of ICDs (last entry in test_data_groundtruth seems to be empty string "")

        # Generated texts to compare to
        corpi = ['../Generated_Text/test_generated_tinyGPT2_nofinetune_withICD.txt',
                '../Generated_Text/test_generated_smallGPT2_nofinetune_withICD.txt',
                '../Generated_Text/test_generated_mediumGPT2_nofinetune_withICD.txt',
                '../Generated_Text/test_generated_tinyGPT2_finetune_withICD.txt',
                '../Generated_Text/test_generated_smallGPT2_finetune_withICD.txt',
                '../Generated_Text/test_generated_mediumGPT2_finetune_withICD.txt']
    elif corpus == 'withDIAGTEXT':
        f.write("Calculating ROUGE for text diagnoses codes appended test set ---------------------------\n")

        # Load data
        test_path_groundtruth = '../Data/test_notes_withDIAGTEXT.txt' #TODO change this funny path name back
        test_data_groundtruth = open(test_path_groundtruth, "r").read().split('<|endoftext|>\n') # list of strings
        test_data_groundtruth = [entry.split("<ICD>")[1] for entry in test_data_groundtruth[:-1]]  # get rid of ICDs (last entry in test_data_groundtruth seems to be empty string "")

        # Generated texts to compare to
        corpi = ['../Generated_Text/test_generated_smallGPT2_finetune_withDIAGTEXT']

    # Calculate ROUGE scores
    for generated_corpus_path in corpi:
        print("Evaluating on generated text file: " + generated_corpus_path + ".......")
        start = time.time()

        if corpus == 'textonly':
            test_data_generated = open(generated_corpus_path, "r").read().split('<|endoftext|>\n')[:-1] # list of strings
        elif corpus == 'withICD':
            test_data_generated = open(generated_corpus_path, "r").read().split('<|endoftext|>\n')[:-1]  # list of strings
            test_data_generated = [entry.split("<ICD>")[1] for entry in test_data_generated]  # get rid of ICDs (last entry in test_data_groundtruth seems to be empty string "")
        elif corpus == 'withDIAGTEXT':
            test_data_generated = open(generated_corpus_path, "r").read().split('<|endoftext|>\n')[:-1]  # list of strings
            test_data_generated = [entry.split("<ICD>")[1] for entry in test_data_generated]
        print("Data loaded")

        # Calculate ROUGE scores
        r = len(test_data_groundtruth)
        rouge1_precision = np.array([])
        rouge1_recall = np.array([])
        rouge1_Fscore = np.array([])

        rouge2_precision = np.array([])
        rouge2_recall = np.array([])
        rouge2_Fscore = np.array([])

        rougeL_precision = np.array([])
        rougeL_recall = np.array([])
        rougeL_Fscore = np.array([])

        for idx in range(r):
            if idx % 100 == 0:
                print("On note number ", idx + 1)

            rouge_scores = scorer.score(test_data_groundtruth[idx], test_data_generated[idx])
            rouge1 = rouge_scores.get('rouge1')
            rouge2 = rouge_scores.get('rouge2')
            rougeL = rouge_scores.get('rougeL')

            rouge1_precision = np.append(rouge1_precision, np.array([rouge1.precision]))
            rouge1_recall = np.append(rouge1_recall, np.array([rouge1.recall]))
            rouge1_Fscore = np.append(rouge1_Fscore, np.array([rouge1.fmeasure]))

            rouge2_precision = np.append(rouge2_precision, np.array([rouge2.precision]))
            rouge2_recall = np.append(rouge2_recall, np.array([rouge2.recall]))
            rouge2_Fscore = np.append(rouge2_Fscore, np.array([rouge2.fmeasure]))

            rougeL_precision = np.append(rougeL_precision, np.array([rougeL.precision]))
            rougeL_recall = np.append(rougeL_recall, np.array([rougeL.recall]))
            rougeL_Fscore = np.append(rougeL_Fscore, np.array([rougeL.fmeasure]))

        f.write("Model: " + generated_corpus_path + "-------------------------\n")
        f.write("ROUGE-1 statistics:\n")
        f.write(".    Precision -- Min: " + str(np.min(rouge1_precision)) + ", 1st Q: " + str(np.quantile(rouge1_precision, 0.25)) +
                                ", Median: " + str(np.quantile(rouge1_precision, 0.5)) + ", Mean: " + str(np.mean(rouge1_precision)) +
                                ", 3rd Q: " + str(np.quantile(rouge1_precision, 0.75)) + ", Max: " + str(np.max(rouge1_precision)) + "\n")
        f.write(".    Recall -- Min: " + str(np.min(rouge1_recall)) + ", 1st Q: " + str(np.quantile(rouge1_recall, 0.25)) +
                             ", Median: " + str(np.quantile(rouge1_recall, 0.5)) + ", Mean: " + str(np.mean(rouge1_recall)) +
                             ", 3rd Q: " + str(np.quantile(rouge1_recall, 0.75)) + ", Max: " + str(np.max(rouge1_recall)) + "\n")
        f.write(".    F1 -- Min: " + str(np.min(rouge1_Fscore)) + ", 1st Q: " + str(np.quantile(rouge1_Fscore, 0.25)) +
                         ", Median: " + str(np.quantile(rouge1_Fscore, 0.5)) + ", Mean: " + str(np.mean(rouge1_Fscore)) +
                         ", 3rd Q: " + str(np.quantile(rouge1_Fscore, 0.75)) + ", Max: " + str(np.max(rouge1_Fscore)) + "\n")

        f.write("ROUGE-2 statistics:\n")
        f.write(".    Precision -- Min: " + str(np.min(rouge2_precision)) + ", 1st Q: " + str(np.quantile(rouge2_precision, 0.25)) +
                                ", Median: " + str(np.quantile(rouge2_precision, 0.5)) + ", Mean: " + str(np.mean(rouge2_precision)) +
                                ", 3rd Q: " + str(np.quantile(rouge2_precision, 0.75)) + ", Max: " + str(np.max(rouge2_precision)) + "\n")
        f.write(".    Recall -- Min: " + str(np.min(rouge2_recall)) + ", 1st Q: " + str(np.quantile(rouge2_recall, 0.25)) +
                             ", Median: " + str(np.quantile(rouge2_recall, 0.5)) + ", Mean: " + str(np.mean(rouge2_recall)) +
                             ", 3rd Q: " + str(np.quantile(rouge2_recall, 0.75)) + ", Max: " + str(np.max(rouge2_recall)) + "\n")
        f.write(".    F1 -- Min: " + str(np.min(rouge2_Fscore)) + ", 1st Q: " + str(np.quantile(rouge2_Fscore, 0.25)) +
                         ", Median: " + str(np.quantile(rouge2_Fscore, 0.5)) + ", Mean: " + str(np.mean(rouge2_Fscore)) +
                         ", 3rd Q: " + str(np.quantile(rouge2_Fscore, 0.75)) + ", Max: " + str(np.max(rouge2_Fscore)) + "\n")

        f.write("ROUGE-L statistics:\n")
        f.write(".    Precision -- Min: " + str(np.min(rougeL_precision)) + ", 1st Q: " + str(np.quantile(rougeL_precision, 0.25)) +
                                ", Median: " + str(np.quantile(rougeL_precision, 0.5)) + ", Mean: " + str(np.mean(rougeL_precision)) +
                                ", 3rd Q: " + str(np.quantile(rougeL_precision, 0.75)) + ", Max: " + str(np.max(rougeL_precision)) + "\n")
        f.write(".    Recall -- Min: " + str(np.min(rougeL_recall)) + ", 1st Q: " + str(np.quantile(rougeL_recall, 0.25)) +
                             ", Median: " + str(np.quantile(rougeL_recall, 0.5)) + ", Mean: " + str(np.mean(rougeL_recall)) +
                             ", 3rd Q: " + str(np.quantile(rougeL_recall, 0.75)) + ", Max: " + str(np.max(rougeL_recall)) + "\n")
        f.write(".    F1 -- Min: " + str(np.min(rougeL_Fscore)) + ", 1st Q: " + str(np.quantile(rougeL_Fscore, 0.25)) +
                         ", Median: " + str(np.quantile(rougeL_Fscore, 0.5)) + ", Mean: " + str(np.mean(rougeL_Fscore)) +
                         ", 3rd Q: " + str(np.quantile(rougeL_Fscore, 0.75)) + ", Max: " + str(np.max(rougeL_Fscore)) + "\n")
        f.write("\n")

        end = time.time()
        print("Time elapsed: ", end - start, " seconds")

f.close()
