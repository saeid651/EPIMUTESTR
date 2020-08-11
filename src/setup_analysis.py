"""
Running EPIMUTESTR

Author: Saeid Parvandeh May 2020
"""
import numpy as np
import pandas as pd
from Helper_Functions import Balance_classes, EA_DMatrix_Germline, EPIMUTESTR

path = '/path_to_the_directory/'
vcf_path = path + 'VCF.gz_file'
index_path = path + 'VCF.gz.tbi_file'

# Create a design matrix
DMatrix = EA_DMatrix_Germline(vcf_path, index_path, way='pEA') # alternatives: pEA, AVG, SUM
DMatrix.to_csv(path + 'DMatrix.csv')
DMatrix = pd.read_csv(path + 'DMatrix.csv', index_col=0)

# Phenotype
pheno_file = pd.read_csv(path + 'phenotype.csv', index_col=0)

# Matching sample ids
common_samples = list(set(pheno_file.index.tolist()).intersection(set(DMatrix.index.tolist())))
DMatrix_fltr = DMatrix.loc[common_samples]
DMatrix_fltr = DMatrix_fltr.reindex(pheno_file.index.tolist())  # Ordering sample ids

# attached labels
class_labels = pheno_file['label_col'].values
DMatrix_fltr['class'] = class_labels

# balance the classes
case_control_idx = Balance_classes(Y=class_labels)
new_DMatrix = DMatrix_fltr.iloc[case_control_idx]

geneList = EPIMUTESTR(X=new_DMatrix, top_features=200, n_cores=2)

fh = open(path + 'output_file.tsv', 'w')
for key, value in geneList.items():
    fh.write(str(key) + '\t' + str(value[0]) + '\t' + str(value[1]) + '\n')
fh.close()
