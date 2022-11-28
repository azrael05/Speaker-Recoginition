# Speaker-Recoginition
This repository contains codes and datasets for Speaker Recognition under Mahadeva Prasanna Sir 

## Pre-requsites
- Python >=3.8
- Libraries - Scikit-learn, librosa, numpy, pandas, seaborn, matplotlib, pickle, glob


## Steps to run the pretrained model (Scripts->.py file)
 1. Clone this repository
 2. Download the Dataset created by IIT-G
 3. Change the path 
 4. Run the File

## Steps to run python code for model (ipynb files)
 1. Clone this repository
 2. Open the required ipynb file
 3. Change base path, train path, test path and model save path (For UBM ubm and adaptation path also)
 4. If using linux, replace "\\" by "/" or as applicable. 

## For X-vector and I-vector steps can be found this <a href="https://github.com/jagabandhumishra/I-MSV-Baseline">repository</a>.

# For the audio dataset 
Directory - Basepath-> Speaker ID -> Files 

## In Files 
- A-D represent 4 session recordings
- Split - Contains each number splitted for Session D
- 3-8 -> Represent the respective length of OTP generated

## Nomenclature
Nomenclature - Gender Speaker_ID Session
- Gender - M-> Male
         <br><t? F -> Female
        
- Speaker_ID -> 001 to 0047 for male
              <br><t> 001 to 003 for female
 
- Session -> A-D representing 4 sessions
