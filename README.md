# ClinicalGPT-2

In this project, we finetune the popular GPT-2 model on MIMIC-III clinical notes to build a language generative model that can produce semi-plausible clinical notes. The objective is to be able to build an automated auto-complete for faster, more efficient, and more personalized real-time clinical visit documentation. Through this, we aim to help reduce physician burnout.

Our models and experiments make use of Hugging Face libraries/blogs/custom Dataset objects and a GPT-2 finetuning tutorial by Phil Schmid. For more information and full credits, please see our writeup. The finetuning data is sampled off of NOTEEVENTS, DIAGNOSES_ICD, ADMISSIONS, and D_ICD_DIAGNOSES data tables from MIMIC-III. We thank all our references for their contributed data, ideas, and guidance.

This project is a collaboration between Jiying Zou (zouj6@gene.com/jiyingzou@gmail.com) and Diego Saldana (diego.saldana@roche.com).

Data citation: MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35.
