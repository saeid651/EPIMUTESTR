"""
This script has a bunch of useful functions for feature selection and classification

Author: Saeid Parvandeh November 2019
"""
import random
import sys
from pysam import VariantFile
import numpy as np
import pandas as pd
import math
from collections import defaultdict, OrderedDict
from scipy.sparse import csr_matrix
from skrebate.turf import TuRF
from skrebate.relieff import ReliefF
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeavePOut
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import statsmodels.stats.multitest as fdr
import scipy
import itertools


def Fetch_Record(vcf_file):
    for rec in vcf_file.fetch():
        if isinstance(rec.info['gene'], tuple):
            EA = rec.info['EA'][0] # controling the isoforms by taking the Canonical transcript (first one)
            GENE = [rec.info['gene'][0]]
        else:
            EA = rec.info['EA'][0]
            GENE = [rec.info['gene']]
        yield GENE, EA, rec.samples

def Adjust_EA(EA):
    try:
        if EA not in ['.', 'silent', 'no_trace', 'no_gene', 'no_action']:
            ea_list = float(EA)
        else:
            ea_list = 0
    except ValueError:
        ea_list = 100
    return ea_list

def Cal_EA(gene_ea, way):
    if not gene_ea:
        ea_out = 0
    elif (way == 'pEA'):
        gene_ea_np = np.array(gene_ea)
        ea_out =  1 - np.prod(1 - gene_ea_np/100)
    elif (way == 'AVG'):
        ea_out = np.mean(gene_ea)  # average EAs
    elif (way == 'SUM'):
        ea_out = np.sum(gene_ea)  # sum EAs (EA burden)
    return ea_out


# fill the matrix by EA score of germline mutations
def EA_DMatrix_Germline(vcf_path, index_path, way = 'pEA'):
    vcf_file = VariantFile(vcf_path,index_filename=index_path)
    # Get sample ids and gene names
    sample_ids = list(vcf_file.header.samples)

    temp = []
    for rec in vcf_file.fetch():
        if isinstance(rec.info['gene'], tuple):
            temp.extend(list(rec.info['gene']))
        else:
            temp.append(rec.info['gene'])
    gene_names = list(set(temp))
    # create an empty matrix of samples by genes
    matrix = np.zeros((len(sample_ids), len(gene_names)), dtype=np.float)
    dmatrix = pd.DataFrame(matrix, columns=gene_names, index=sample_ids)
    EA_collections = defaultdict(dict)
    for GENE, EA, SAMPLE in Fetch_Record(vcf_file):
        for gene in gene_names:
            if gene in GENE:
                for sample in sample_ids:
                    try:
                        if sum(SAMPLE[sample].get('GT')) == 1:
                            try:
                                EA_collections[sample][gene] += [Adjust_EA(EA)]
                            except KeyError:
                                EA_collections[sample][gene] = [Adjust_EA(EA)]
                        elif sum(SAMPLE[sample].get('GT')) == 2:
                            try:
                                EA_collections[sample][gene] += [Adjust_EA(EA)] * 2
                            except:
                                EA_collections[sample][gene] = [Adjust_EA(EA)] * 2
                    except TypeError:
                        pass
    for gene in gene_names:
        for sample in sample_ids:
            if gene in EA_collections[sample].keys():
                gene_ea = EA_collections[sample][gene]
                # To deal with list of lists
                gene_ea_merged = list(pd.core.common.flatten(gene_ea))
                gene_ea_merged = [i for i in gene_ea_merged if i != 0]
                dmatrix.loc[sample, gene] = Cal_EA(gene_ea_merged, way)
    return dmatrix

# fill the matrix by EA score of somatic mutations
def EA_DMatrix_Somatic(vcf_path, index_path):
    vcf_file = VariantFile(vcf_path,index_filename=index_path)
    # Get sample ids and gene names
    sample_ids = list(vcf_file.header.samples)

    temp = []
    for rec in vcf_file.fetch():
        if isinstance(rec.info['gene'], tuple):
            temp.extend(list(rec.info['gene']))
        else:
            temp.append(rec.info['gene'])
    gene_names = list(set(temp))
    # create an empty matrix of samples by genes
    matrix = np.zeros((len(sample_ids), len(gene_names)), dtype=np.float)
    dmatrix = pd.DataFrame(matrix, columns=gene_names, index=sample_ids)
    for GENE, EA, SAMPLE in Fetch_Record(vcf_file):
        for gene in gene_names:
            if gene in GENE:
                for sample in sample_ids:
                    try:
                        if sum(SAMPLE[sample].get('GT')) in [1, 2]:
                            try:
                                dmatrix.loc[sample, gene] = max(dmatrix.loc[sample, gene], Adjust_EA(EA))
                            except ValueError:
                                dmatrix.loc[sample, gene] = 100
                    except TypeError:
                        pass
    return dmatrix


def GMM(Y, n_components=2):
    Y = Y.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full').fit(Y)
    weights = gmm.weights_
    means = gmm.means_
    covars = gmm.covariances_

    gmm_probs = gmm.predict_proba(Y)
    max_post_idx = np.argmax(gmm_probs, axis=1)
    cutoff_points = []
    for component in range(n_components):
        group_idx = max_post_idx == component
        cutoff_points.append([float(max(Y[group_idx]))])
    # return weights, means, covars, cutoff_points
    class_labels = np.where(Y < min(cutoff_points), 1, 0)
    return weights, means, covars, class_labels, cutoff_points

def Balance_classes(Y):
    cases_idx = np.where(Y == 1)[0].tolist()
    controls_idx = np.where(Y == 0)[0].tolist()
    min_idx = np.argmin((len(cases_idx), len(controls_idx)))
    smaller_group_idx, larger_group_idx = (cases_idx, controls_idx) if min_idx == 0 else (controls_idx, cases_idx)
    larger_sub_idx = random.sample(larger_group_idx, len(smaller_group_idx))
    new_class_idx = sorted(smaller_group_idx + larger_sub_idx)
    return new_class_idx


def EPIMUTESTR(X=None, top_features=200, nn=None, discrete_threshold=10, verbose=False, n_cores=1, estimator='relief', pct=0.5):
    X = X.loc[:, (X != 0).any(axis=0)]
    features, labels = X.drop('class', axis=1).values, X['class'].values
    features = np.nan_to_num(features)
    headers = list(X.drop("class", axis=1))
    if nn == None:
        nn = math.floor(0.154 * (X.shape[1] - 1))
    if (estimator=='TuRF'):
        # Total Unduplicated Reach and Frequency (TURF)
        fs = TuRF(core_algorithm="ReliefF", n_features_to_select=top_features, n_neighbors=nn, pct=pct, verbose=verbose, n_jobs=n_cores)
        fs.fit(features, labels, headers)
    elif (estimator == 'relief'):
        # ReliefF stand alone
        fs = ReliefF(n_features_to_select=top_features, n_neighbors=nn, discrete_threshold=discrete_threshold, verbose=verbose, n_jobs=n_cores)
        fs.fit(features, labels)

    scoreDict = dict(zip(X.drop('class', axis=1).columns, fs.feature_importances_))
    scoreDict_sorted = {i[1]: i[0] for i in sorted(zip(scoreDict.values(), scoreDict.keys()), reverse=True)}
    scores_list = list(scoreDict_sorted.values())
    pos_scores_list = [n for n in scores_list if n > 0]
    # calculate the P value and adjusted P value
    gene_scores = np.sqrt(pos_scores_list)
    gene_scores_mean = np.mean(gene_scores)
    gene_scores_sd = np.std(gene_scores)
    pvals = []
    for score in gene_scores:
        pvals.append(scipy.stats.norm(gene_scores_mean, gene_scores_sd).sf(score))
    # Benjamini/Hachberg FDR correction
    qvals = fdr.multipletests(np.asarray(pvals), method='fdr_bh', is_sorted=True)

    geneList = dict(zip(scoreDict_sorted.keys(), zip(pvals, qvals[1])))

    return geneList

def adj_norm(X=None): #, Y=None
    # Normalize each column to 0-1
    X_features = X.drop('class', axis = 1)
    X_Maxs = X_features.max()
    X_norm = X_features / max(X_Maxs)  # divide each column by max
    X_norm['class'] = X['class'].values

    return X_norm

def classification(X_train=None, Y_train=None, X_test=None, Y_test=None):

    # ElasticNet
    # Note: Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on features with approximately the same scale.
    # You can preprocess the data with a scaler from sklearn.preprocessing.
    enet_model = LogisticRegressionCV(penalty='elasticnet', solver="saga",
                                      l1_ratios=np.arange(0, 1, 0.1).tolist(),
                                      cv=10)  # Stochastic Average Gradient descent solver.
    y_test_pred_enet = enet_model.fit(X_train, Y_train).predict(X_test)
    enet_acc = balanced_accuracy_score(Y_test, list(y_test_pred_enet))
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=1000, oob_score=True).fit(X_train, Y_train)
    y_test_pred_rf = rf_model.predict(X_test)
    rf_acc = balanced_accuracy_score(Y_test, list(y_test_pred_rf))
    # Decision Tree
    dt_model = DecisionTreeClassifier().fit(X_train, Y_train)
    y_test_pred_dt = dt_model.predict(X_test)
    dt_acc = balanced_accuracy_score(Y_test, list(y_test_pred_dt))
    # SVM
    svm_model = SVC().fit(X_train, Y_train)
    y_test_pred_svm = svm_model.predict(X_test)
    svm_acc = balanced_accuracy_score(Y_test, list(y_test_pred_svm))
    # Naive Bayes
    nb_model = GaussianNB().fit(X_train, Y_train)
    y_test_pred_nb = nb_model.predict(X_test)
    nb_acc = balanced_accuracy_score(Y_test, list(y_test_pred_nb))

    return enet_acc, rf_acc, dt_acc, svm_acc, nb_acc

def cnCV(X_train=None, X_validation=None, top_features=200, k=[10,10], cv_type='cncv', n_cores = 1, pct = 0.5, normalize=False):
    # Get the labels to create folds
    if isinstance(X_train, list):
        XA = X_train[0]
        labels = XA['class'].values
    else:
        sys.exit('Input should be a list of dataframe(s)')

    models_dic = {1:'enet', 2:'rf', 3:'dt', 4:'svm', 5:'nb'}

    # Normalize training dataset(s)
    if normalize==True:
        for i in range(len(X_train)):
            XA = X_train[i]
            X_train[i] = adj_norm(X=XA)

    if cv_type == 'LOO_cncv':
        # Leave-one-out cross validation
        nCV_folds = LeaveOneOut()
    elif cv_type == 'cncv':
        # k-folds cross validation
        nCV_folds = StratifiedKFold(n_splits=k[0])

    # Create outer-folds
    for train_outer_idx, test_outer_idx in nCV_folds.split(np.zeros(len(labels)), labels):
        Xs_train_outer = []
        Xs_test_outer = []
        for idx in range(len(X_train)):
            X_train_fltr = X_train[idx].drop('class', axis = 1)
            Xs_train_outer.append(X_train_fltr.iloc[train_outer_idx, :])
            Xs_test_outer.append(X_train_fltr.iloc[test_outer_idx, :])
        train_outer_res = X_train[0]['class'].values[train_outer_idx]
        test_outer_res = X_train[0]['class'].values[test_outer_idx]
        # Create k-inner-folds
        inner_genes_list = defaultdict(list)
        models_acc = defaultdict(list)
        kfolds = StratifiedKFold(n_splits=k[1])
        for train_inner_idx, test_inner_idx in kfolds.split(np.zeros(len(train_outer_res)), train_outer_res):
            Xs_train_inner = []
            Xs_test_inner = []
            for idx in range(len(Xs_train_outer)):
                # Break down list of dfs
                XAs_train_outer = Xs_train_outer[idx]
                # create inner train
                X_train_inner = XAs_train_outer.iloc[train_inner_idx, :]
                X_test_inner = XAs_train_outer.iloc[test_inner_idx, :]
                train_inner_res = train_outer_res[train_inner_idx]
                test_inner_res = train_outer_res[test_inner_idx]
                X_train_inner['class'] = train_inner_res
                # call ReliefF learner to score the genes
                scored_genes = EPIMUTESTR(X=X_train_inner, top_features=top_features, n_cores=n_cores, pct=pct)
                # genes_dic = call a function to get top genes or based on cdf p-value # select top genes
                genes_dic = list(scored_genes.keys())[:top_features]

                Xs_train_inner.append(X_train_inner.loc[:, genes_dic])
                Xs_test_inner.append(X_test_inner.loc[:, genes_dic])

                if len(Xs_train_inner) > 1:
                    # data integration
                    integrated_train_inner_df = pd.concat(Xs_train_inner, axis=1, sort=False)
                    integrated_test_inner_df = pd.concat(Xs_test_inner, axis=1, sort=False)
                    integrated_train_inner_df['class'] = train_inner_res
                    integrated_test_inner_df['class'] = test_inner_res

                    # run the EPIMUTESTR on integrated data
                    scored_genes = EPIMUTESTR(X=integrated_train_inner_df, top_features=top_features, n_cores=n_cores)
                    top_genes = list(scored_genes.keys())[:top_features]
                    inner_genes_list['integrated_top_genes'].append(top_genes)
                else:
                    inner_genes_list['single_top_genes'].append(genes_dic)

        if len(X_train) > 1:
            # data integration
            integrated_train_outer_df = pd.concat(Xs_train_outer, axis=1, sort=False)
            integrated_test_outer_df = pd.concat(Xs_test_outer, axis=1, sort=False)

            # selecting the features from cv
            list_of_features = inner_genes_list['integrated_top_genes']
            features = list(set(np.concatenate(list_of_features).tolist()))
            train_outer_fltr = integrated_train_outer_df.loc[:, features]
            test_outer_fltr = integrated_test_outer_df.loc[:, features]
            train_outer_fltr['class'] = train_outer_res
            test_outer_fltr['class'] = test_outer_res
        else:
            features = inner_genes_list['single_top_genes'][0]
            train_outer_fltr = Xs_train_outer[0].loc[:, features]
            test_outer_fltr = Xs_test_outer[0].loc[:, features]
            train_outer_fltr['class'] = train_outer_res
            test_outer_fltr['class'] = test_outer_res

        X_train_outer_features, Y_train_outer = train_outer_fltr.drop('class', axis=1).values, \
                                            train_outer_fltr['class'].values
        X_test_outer_features, Y_test_outer = test_outer_fltr.drop('class', axis=1).values, \
                                          test_outer_fltr['class'].values

        # Replace nans
        X_train_outer_features = np.nan_to_num(X_train_outer_features)
        X_test_outer_features = np.nan_to_num(X_test_outer_features)

        # Call stacking function to compute the accuracy of each fold
        enet_acc, rf_acc, dt_acc, svm_acc, nb_acc = classification(X_train=X_train_outer_features,
                                                                    Y_train=Y_train_outer,
                                                                    X_test=X_test_outer_features,
                                                                    Y_test=Y_test_outer)

        models_acc['enet'].extend([enet_acc])
        models_acc['rf'].extend([rf_acc])
        models_acc['dt'].extend([dt_acc])
        models_acc['svm'].extend([svm_acc])
        models_acc['nb'].extend([nb_acc])

        for model in list(models_dic.values()):
            avg_acc = np.mean(models_acc[model])
            models_acc['avg'].append(avg_acc)

        max_acc_idx = models_acc['avg'].index(max(models_acc['avg']))
        final_model_name = list(models_dic.values())[max_acc_idx]

    # consensus features
    if len(X_train) > 1:
        consensus_genes = list(set.intersection(*map(set, inner_genes_list['integrated_top_genes'])))
    else:
        consensus_genes = list(set.intersection(*map(set, inner_genes_list['single_top_genes'])))

    # define variables
    train_acc = None
    valid_acc = None
    if (X_validation != None):
        if len(X_train) > 1:
            # data integration
            integrated_train = pd.concat(X_train, axis=1, sort=False)
            integrated_valid = pd.concat(X_validation, axis=1, sort=False)
            integrated_train_data = integrated_train.loc[:, consensus_genes]
            integrated_valid_data = integrated_valid.loc[:, consensus_genes]
            integrated_train_data['class'] = X_train[0]['class'].values
            integrated_valid_data['class'] = X_validation[0]['class'].values
        else:
            XA_train = X_train[0]
            XA_valid = X_validation[0]
            integrated_train_data = XA_train.loc[:, consensus_genes]
            integrated_valid_data = XA_valid.loc[:, consensus_genes]
            integrated_train_data['class'] = X_train[0]['class'].values
            integrated_valid_data['class'] = X_validation[0]['class'].values

        X_train_features, Y_train = integrated_train_data.drop('class', axis=1).values, integrated_train_data[
            'class'].values
        X_valid_features, Y_valid = integrated_valid_data.drop('class', axis=1).values, integrated_valid_data[
            'class'].values
        # Replace nans
        X_train_features = np.nan_to_num(X_train_features)
        X_valid_features = np.nan_to_num(X_valid_features)
        # # convert to sparse
        # X_train_sparse = csr_matrix(X_train_features)
        # X_valid_sparse = csr_matrix(X_valid_features)
        if final_model_name == 'enet':
            enet_model = LogisticRegressionCV(penalty='elasticnet', solver="saga",
                                              l1_ratios=np.arange(0, 1, 0.1).tolist(),
                                              cv=10)  # Stochastic Average Gradient descent solver.
            y_train_pred = enet_model.fit(X_train_features, Y_train).predict(X_train_features)
            y_valid_pred = enet_model.fit(X_train_features, Y_train).predict(X_valid_features)
        elif final_model_name == 'rf':
            # Random Forest
            rf_model = RandomForestClassifier(n_estimators=1000, oob_score=True).fit(X_train_features, labels)
            y_train_pred = rf_model.predict(X_train_features)
            y_valid_pred = rf_model.predict(X_valid_features)
        elif final_model_name == 'dt':
            # Decision Tree
            dt_model = DecisionTreeClassifier().fit(X_train_features, Y_train)
            y_train_pred = dt_model.predict(X_train_features)
            y_valid_pred = dt_model.predict(X_valid_features)
        elif final_model_name == 'svm':
            # SVM
            svm_model = SVC().fit(X_train_features, Y_train)
            y_train_pred = svm_model.predict(X_train_features)
            y_valid_pred = svm_model.predict(X_valid_features)
        elif final_model_name == 'nb':
            # Naive Bayes
            nb_model = GaussianNB().fit(X_train_features, Y_train)
            y_train_pred = nb_model.predict(X_train_features)
            y_valid_pred = nb_model.predict(X_valid_features)

        train_acc = balanced_accuracy_score(Y_train, list(y_train_pred))
        valid_acc = balanced_accuracy_score(Y_valid, list(y_valid_pred))

    return {'Training_Accuracy': train_acc, 'Validation_Accuracy': valid_acc, 'Consensus_Features': consensus_genes}
