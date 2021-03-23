'''
File to prepare data in necessary formats for finetuning
Gets the following corpora into list-of-string format:
  - Note text only
  - ICD appended to note text
  - Diagnoses text appended to note text
This code is mostly original, with some guidance for the SQL portion from
https://stackoverflow.com/questions/30627968/merge-pandas-dataframes-where-one-value-is-between-two-others
'''
import sqlite3
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split

########################
### DATA PREPARATION ###
########################

# Read clinical notes data into pandas dataframe
num_notes = int(5000/0.7) #training has ~5k notes
notes = pd.read_csv("~/Documents/CS224N/MIMIC/mimic-iii-clinical-database-1.4/NOTEEVENTS.csv", nrows = num_notes)
icd = pd.read_csv("~/Documents/CS224N/MIMIC/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv")
admissions = pd.read_csv("~/Documents/CS224N/MIMIC/mimic-iii-clinical-database-1.4/ADMISSIONS.csv")
icd_map = pd.read_csv("~/Documents/CS224N/MIMIC/mimic-iii-clinical-database-1.4/D_ICD_DIAGNOSES.csv")

# Data wrangling to get ethnicity and most recent ICD codes for each note
icd = icd.merge(icd_map, how = 'left', on = ['ICD9_CODE'])
icd_grouped = icd.groupby(['SUBJECT_ID','HADM_ID']).agg({'ICD9_CODE': lambda x: ','.join(str(code) for code in x),
                                                         'LONG_TITLE': lambda x: ','.join(str(title) for title in x)})
admissions = admissions.merge(icd_grouped, how = 'left', on = ['SUBJECT_ID','HADM_ID'])
notes = notes.merge(admissions[['HADM_ID','ADMITTIME','DISCHTIME']], how = 'inner', on = 'HADM_ID').drop(columns=['ROW_ID']).drop_duplicates() #removed 5 dupes

# -------- Code skeleton for following SQL connection part thanks to StackOverflow post, linked in description above
# Make the DB in memory
conn = sqlite3.connect(':memory:')

# Write the tables
notes.to_sql('notes', conn, index=False)
admissions.to_sql('admissions', conn, index=False)

qry = '''
    --This code below is my implementation
    SELECT SUBJECT_ID, 
           HADM_ID,
           CATEGORY,
           DESCRIPTION,
           TEXT,
           ETHNICITY,
           GROUP_CONCAT(PAST_ICDS) AS PAST_ICDS,  --from most to least recent
           GROUP_CONCAT(PAST_DIAGNOSES) AS PAST_DIAGNOSES --from most to least recent
    FROM (
      SELECT  
          a.SUBJECT_ID,
          a.HADM_ID,
          a.CATEGORY,
          a.DESCRIPTION,
          a.TEXT,
          b.ETHNICITY,
          b.ICD9_CODE AS PAST_ICDS,
          b.LONG_TITLE AS PAST_DIAGNOSES
      FROM notes a
      LEFT JOIN admissions b
        ON a.SUBJECT_ID == b.SUBJECT_ID
        AND b.ADMITTIME BETWEEN DATE(a.ADMITTIME,'-1 year') AND DATE(a.DISCHTIME)
      ORDER BY DATE(b.ADMITTIME) DESC             --from most to least recent
          )
    GROUP BY 1,2,3,4,5,6
    '''
notes_full = pd.read_sql_query(qry, conn) #not using this until experiments
# --------

# Train-validation-test split (70-20-10%)
train, val = train_test_split(notes_full, test_size=0.30, random_state=42)
val, test = train_test_split(val, test_size=0.33, random_state=88)

print("Training set size:", len(train))
print("Validation set size:", len(val))
print("Test set size:", len(test))


############################
### DATA WITH NOTES ONLY ###
############################

#Concatenate the notes together with <|endoftext|> tokens between them, save to .txt file
def concat_notes(note_list, filename):
  result = reduce(lambda x, y : ''.join((x, '<|endoftext|>\n', y)), note_list)
  write_path = '../Data/%s.txt' % filename
  with open(write_path, "w") as text_file:
    text_file.write(''.join(result))

concat_notes(train["TEXT"], filename = "train_notes_textonly")
print("Done with training notes")
concat_notes(val["TEXT"], filename = "val_notes_textonly")
print("Done with validation notes")
concat_notes(test["TEXT"], filename = "test_notes_textonly")
print("Done with testing notes")


#############################
### DATA WITH ICD + NOTES ###
#############################

#Concatenate the notes together with <|endoftext|> tokens between them, save to .txt file
def concat_notes_withICD(note_df, filename):
  note_df = note_df.reset_index()
  concatenated = ""
  for i in range(note_df.shape[0]):
    icds = note_df['PAST_ICDS'][i]
    text = note_df['TEXT'][i]
    concatenated += " ".join(str(icds).split(",")) + '<ICD>' + str(text) + '<|endoftext|>\n'
  write_path = '../Data/%s.txt' % filename
  with open(write_path, "w") as text_file:
    text_file.write(concatenated)

concat_notes_withICD(train, filename = "train_notes_withICD")
print("Done with training notes + ICD")
concat_notes_withICD(val, filename = "val_notes_withICD")
print("Done with validation notes + ICD")
concat_notes_withICD(test, filename = "test_notes_withICD")
print("Done with testing notes + ICD")


########################################
### DATA WITH TEXT DIAGNOSES + NOTES ###
########################################

#Concatenate the notes together with <|endoftext|> tokens between them, save to .txt file
def concat_notes_withDIAGTEXT(note_df, filename):
  note_df = note_df.reset_index()
  concatenated = ""
  for i in range(note_df.shape[0]):
    icds = note_df['PAST_DIAGNOSES'][i]
    text = note_df['TEXT'][i]
    concatenated += " ".join(str(icds).split(",")) + '<ICD>' + str(text) + '<|endoftext|>\n'
  write_path = '../Data/%s.txt' % filename
  with open(write_path, "w") as text_file:
    text_file.write(concatenated)

concat_notes_withDIAGTEXT(train, filename = "train_notes_withDIAGTEXT")
print("Done with training notes + diagnoses text")
concat_notes_withDIAGTEXT(val, filename = "val_notes_withDIAGTEXT")
print("Done with validation notes + diagnoses text")
concat_notes_withDIAGTEXT(test, filename = "test_notes_withDIAGTEXT")
print("Done with testing notes + diagnoses text")