# EMBC2024_FingerTappingSemiSupervised
Official repository for IEEE EMBC 2024 Submission "Enhancing Model Generalizability In Parkinson’s Disease Automatic Assessment: A Semi-Supervised Approach Across Independent Experiments"

# Code #
The experiment can be reproduced by running the .py file in code folder. Before running, setup your own wandb project and replace the name of such project in the code by replacing the string "Your project name". In this way all the run of the experiment will be automatically logged and can be compared online through the wandb interface.

# Data #

Data folder contains the .csv where the data from PARK, AAP, PDMOT datasets are stored all together. This file is read at the begigging of the .py script. Data made available are the features obtained by applying the feature extraction pipeline proposed for PARK dataset to AAP and PDMOT datasets as well. For AAP and PDMOT, a subject identifier is not available, so the "filename" column is just a placeholder. The label "score" is the clinical score assigned by the expert raters for PDMOT and PARK. For AAP the clinical scores are not available, value 1 indicates only a diagnosis of Parkinson's and should not be considered as a UPDRS score. The "dataset" column represents PARK (value 1), PDMOT (value 2) and AAP (value 3). 

# References #
Data for PARK and AAP were taken from two other original works. If you re-use this code, please cite their works as well.

PARK:
Islam, M.S., Rahman, W., Abdelkader, A. et al. 
Using AI to measure Parkinson’s disease severity at home. npj Digit. Med. 6, 156 (2023). https://doi.org/10.1038/s41746-023-00905-9

PDMOT:
Yang, Ning, et al. 
"Automatic detection pipeline for accessing the motor severity of parkinson’s disease in finger tapping and postural stability." IEEE Access 10 (2022): 66961-66973.


