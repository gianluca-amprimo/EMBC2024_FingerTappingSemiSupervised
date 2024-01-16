###IMPORTANT NOTE: this code incorporate is inspired from the work in https://github.com/ROC-HCI/finger-tapping-severity. 
###If you employ this code for your project, please cite also their work. 

import time
from sklearn.metrics import confusion_matrix
import numpy as onp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import psutil
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.semi_supervised import SelfTrainingClassifier
import plotly
import wandb
import random
from imblearn.metrics import specificity_score
from pingouin import intraclass_corr
from krippendorff import alpha
from tqdm import tqdm
from pandas import DataFrame
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, cohen_kappa_score, r2_score, \
    mean_absolute_percentage_error, accuracy_score, f1_score, precision_score, recall_score
from copy import deepcopy

RATING_FILE = "../data/complete_dataset_nofilt.csv"
output_path="../outputs/"
figure_path="../outputs/figures/"

def acceptable_accuracy(y_true, y_pred, tolerance=1):
    """
    Computes acceptable accuracy by considering predictions within a range of ±tolerance error as correct.

    Parameters:
    - y_true: The true labels.
    - y_pred: The predicted labels.
    - tolerance: The tolerance for acceptable accuracy (default is 1).

    Returns:
    - Acceptable accuracy.
    """
    correct = 0
    total = len(y_true)

    for true_label, pred_label in zip(y_true, y_pred):
        if abs(true_label - pred_label) <= tolerance:
            correct += 1

    acc = correct / total
    return acc

def load():
    df = pd.read_csv(RATING_FILE, sep=';')
    df=df.dropna()
    features = df.drop(['label', 'dataset', 'filename'], axis='columns')
    labels = df["label"]
    domain_labels=df['dataset']

    def parse(name:str):
        if name=="noloso": return name
        if name.startswith("NIH"): [ID, *_] = name.split("-")
        else: [*_, ID, _, _] = name.split("-")
        return ID
    
    df["id"] = df.filename.apply(parse) #da reinserire per provare eventualmente la loso solo sul dataset npj,

    return features, labels, domain_labels, df["id"]

def metrics(preds, labels):
    results = {}

    preds, labels = onp.array(preds), onp.array(labels)
    results["mae"] = mean_absolute_error(labels, preds)
    results["mse"] = mean_squared_error(labels, preds.T)
    results["r2"] = r2_score(labels, preds)
    results["mape"] = mean_absolute_percentage_error(labels + 1, preds + 1) # shift labels by 1
    results["pearsonr"]= onp.corrcoef(labels, preds.T)[0,1]

    rounded_preds = onp.round(preds)
    rounded_labels = onp.round(labels)
    results["kappa.no.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights=None)
    results["kappa.linear.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="linear")
    results["kappa.quadratic.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="quadratic")
    results["accuracy"] = accuracy_score(rounded_labels, rounded_preds)
    results["acceptable_accuracy"]=acceptable_accuracy(rounded_labels, rounded_preds)
    results["f-1 score"]=f1_score(rounded_labels, rounded_preds, average='micro')
    results["precision"]=precision_score(rounded_labels, rounded_preds, average='micro')
    results["recall"]=recall_score(rounded_labels, rounded_preds, average='micro')
    results["specificity"] = specificity_score(rounded_labels, rounded_preds, average='micro')
    results["kenndalltau"], _ = stats.kendalltau(labels, preds)
    results["spearmanr"], _ = stats.spearmanr(labels, preds)

    return results

def shallowModel(**cfg):

    if cfg["model"] == "LGBMClassifier":
        return LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
            verbose=-1,
            n_jobs=8
        )
      
    raise ValueError("Unknown model")

def semisupervision_vs_supervision(features, labels, domain_labels, mode, cfg, loso_identity):

    to_select=['wrist_mvmnt_x_median', 'frequency_quartile_range_denoised',
           'frequency_stdev_denoised', 'frequency_lr_fitness_r2_denoised',
           'acceleration_min_denoised', 'amplitude_median_denoised',
           'amplitude_quartile_range_denoised', 'amplitude_max_denoised',
           'amplitude_stdev_denoised', 'amplitude_entropy_denoised',
           'amplitude_stdev_trimmed', 'amplitude_decrement_slope_denoised',
           'amplitude_decrement_end_to_mean_denoised',
           'amplitude_decrement_slope_trimmed', 'speed_quartile_range_denoised',
           'speed_median_trimmed', 'period_quartile_range_trimmed',
           'period_min_denoised', 'frequency_lr_slope_trimmed',
           'maxFreezeDuration_trimmed', 'maxFreezeDuration_denoised',
           'wrist_mvmnt_y_max']    #relevant features in the original paper "Using AI to measure Parkinson’s disease severity at home" for PARK dataset

    PARK_labels = labels[domain_labels == 1]
    PARK_data = features.loc[domain_labels == 1]
    loso_identity = loso_identity.loc[domain_labels == 1]
    PARK_data_red=PARK_data[to_select]

    PDMOT_labels = labels[domain_labels == 2]
    PDMOT_data = features[domain_labels == 2]
    PDMOT_data_PARK=PDMOT_data[PARK_data_red.columns]


    aap_labels = labels[domain_labels == 3]
    aap_data = features.loc[domain_labels == 3]
    aap_data_PARK = aap_data[PARK_data_red.columns]

    all_labels = []
    all_preds = []
    all_preds_selftrained = []
    all_preds_PARKPDMOT=[]
    all_preds_labelprop=[]
    all_preds_labelspread=[]


    cv=StratifiedKFold(n_splits=abs(cfg['cv']), shuffle=True, random_state=42)

    i = 0
    if cfg['name']=='baseline':
        for splt1, splt2 in tqdm(cv.split(PDMOT_data_PARK, PDMOT_labels)): #test_index, train_index
            i += 1
            if cfg['cv'] >0:
                test_index=splt1
                train_index=splt2
            else:
                train_index=splt1
                test_index=splt2

            regressor_PARK_PDMOT = shallowModel(**cfg)
            regressor_PARK = shallowModel(**cfg)

            regressor_PARK_PDMOT.fit(pd.concat([PARK_data_red, PDMOT_data_PARK.iloc[train_index]]),
                                         pd.concat([PARK_labels, PDMOT_labels.iloc[train_index]]))
            regressor_PARK.fit(PARK_data_red, PARK_labels)

            y_pred = regressor_PARK.predict(PDMOT_data_PARK.iloc[test_index])
            y_pred_PARKPDMOT = regressor_PARK_PDMOT.predict(PDMOT_data_PARK.iloc[test_index])

            all_labels.extend(PDMOT_labels.iloc[test_index])
            all_preds_PARKPDMOT.extend(y_pred_PARKPDMOT)
            all_preds.extend(y_pred)

        wandb.init(project="Your project name", config=cfg, name=cfg['name']+"_PARK", group=cfg['group'], reinit=True,
                   notes="PARK only supervised classifier")
        print("PARK only:")
        results = metrics(all_preds, all_labels)
        # Compute the confusion matrix

        cm = confusion_matrix(all_labels, all_preds)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[0,1,2,3,4],
                    yticklabels=[0,1,2,3,4])
        plt.xlabel('Predicted UPDRS score')
        plt.ylabel('Groundtruth scoring')
        plt.title('Confusion Matrix')
        plt.show()

        wandb.log(results)
        print(results)

        wandb.init(project="Your project name", config=cfg, name=cfg['name']+"_PARK_and_PDMOT", group=cfg['group'],
                   reinit=True,
                   notes="PARK and PDMOT supervised classifier")
        print("PARK and PDMOT only:")
        results = metrics(all_preds_PARKPDMOT, all_labels)
        wandb.log(results)
        print(results)
        return



    wandb.init(project="Your project name", config=cfg, name=cfg['name'], group=cfg['group'], reinit=True,
               notes=f'Gridsearch for {cfg["name"].split("_")[0]}')

    all_labels=[]
    i=0
    reached_max_iter=0
    for splt1, splt2 in tqdm(cv.split(PDMOT_data_PARK, PDMOT_labels)):
        i += 1
        if cfg['cv'] > 0:
            test_index = splt1
            train_index = splt2
        else:
            train_index = splt1
            test_index = splt2

        if 'self' in cfg['name']:
             regressor_selftrained = SelfTrainingClassifier(shallowModel(**cfg), threshold=cfg['threshold'], criterion=cfg['method'], k_best=cfg['k_best'],  max_iter=cfg['max_iter'], verbose=True)     
        
        if 'aap' in cfg['name']:
            feat_train = pd.concat([PARK_data_red, aap_data_PARK])
            label_train = pd.concat([PARK_labels, -aap_labels])
        elif 'mot' in cfg['name']:
            feat_train = pd.concat([PARK_data_red, PDMOT_data_PARK.iloc[train_index]])
            label_train = pd.concat([PARK_labels, pd.Series(-onp.ones(train_index.shape), index=train_index)])
        else:
            feat_train = pd.concat([PARK_data_red, aap_data_PARK, PDMOT_data_PARK.iloc[train_index]])
            label_train = pd.concat([PARK_labels, -aap_labels, pd.Series(-onp.ones(train_index.shape), index=train_index)])

        feat_test = PDMOT_data_PARK.iloc[test_index]
        label_test = PDMOT_labels.iloc[test_index]

        X_train_semisup, X_test = feat_train.to_numpy(), feat_test.to_numpy()
        y_train_semisup, y_test = label_train.to_numpy(), label_test.to_numpy()

        if 'self' in cfg['name']:
            regressor_selftrained.fit(X_train_semisup, y_train_semisup)
            y_pred_selftrained = regressor_selftrained.predict(X_test)
            all_preds_selftrained.extend(y_pred_selftrained)
            labels_self= pd.Series(regressor_selftrained.transduction_[regressor_selftrained.labeled_iter_>1], index=label_train.index[regressor_selftrained.labeled_iter_>1])
            if regressor_selftrained.termiPARKion_condition_=='max_iter':
                reached_max_iter+=1
            data = labels_self[labels_self.index.isin(aap_labels.index)]
        
        if 'mot' not in cfg['name']:
            plt.figure()
            sns.set_style("whitegrid")
            sns.kdeplot(data=data, label="Label assigned",
                        common_norm=False)
            plt.title("Label assigned for AAP")
            plt.legend()
            wandb.log({"Label assigned to AAP": plt})
            plt.close()

        if 'aap' not in cfg['name']:
            plt.figure()
            sns.set_style("whitegrid")
            plt.title("Label assigned for PDMOT")

            sns.kdeplot(data=PDMOT_labels.iloc[train_index], label="Groundtruth", common_norm=False, linestyle='--', color='black')
            if 'self' in cfg['name']:
                sns.kdeplot(data=labels_self[labels_self.index.isin(PDMOT_labels.index)], label="Label selftrained",
                            common_norm=False)
            plt.legend()
            wandb.log({"Label assigned to PDMOT":plt})
            plt.close()
            #plt.show()

        all_labels.extend(y_test)

    if 'self' in cfg['name']:
        print("Selftrained:")
        results = metrics(all_preds_selftrained, all_labels)
        cm = confusion_matrix(all_labels, all_preds_selftrained)
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=[0, 1, 2, 3, 4],
                    yticklabels=[0, 1, 2, 3, 4])
        plt.xlabel('Predicted UPDRS score')
        plt.ylabel('Groundtruth scoring')
        plt.title('Confusion Matrix')
        plt.show()
        wandb.log(results)
        print(results)
        return reached_max_iter
  


def main(cfg):

    features, labels, domain_labels, loso_labels = load()
    ret= semisupervision_vs_supervision(features, labels, domain_labels, cfg['val_mode'], cfg, loso_labels)
    return ret


if __name__ == "__main__":

    param_grid_self = [
        {'name': "self_training",
        'group': "semisupervision_cv",
        'method': 'threshold',
        'threshold': [0.80, 0.85,  0.90, 0.95, 0.97],
        'max_iter': [15, 20, 25, 30],
        'k_best':1,
        'model': "LGBMClassifier",
        'random_state': 42,
        'seed': 42,
        'learning_rate': 0.013,
        'n_estimators': 611,
        'max_depth': 3,
        'subsample': 0.8
    },
        {'name': "self_training",
        'group': "semisupervision_cv",
        'method': 'k_best',
        'threshold': 0.99,
        'k_best': [10, 50, 100, 150, 200, 300],
        'max_iter':[30],
        'model': "LGBMClassifier",
        'random_state': 42,
        'seed': 42,
        'learning_rate': 0.013,
        'n_estimators': 611,
        'max_depth': 3,
        'subsample': 0.8
    }]

    param_grid_baseline={'name': "baseline",
            'group': "semisupervision_cv",
            'method': 'knn',
            'n_neighbors': range(1, 50, 2),
            'model': "LGBMClassifier",
            'val_mode': 'loso',
            'scaling_method': "StandardScaler",
            'random_state': 42,
            'seed': 42,
            'n': 22,
            'learning_rate': 0.013,
            'n_estimators': 611,
            'max_depth': 3,
            'subsample': 0.8}


    group="semisupervision_cv"
    for cv in [10, 5, 3, 2, -3, -5]:# 10%, 20%, 30%, 50%, 66,6%, 80%
        if cv!=5:
            perc= 100/cv if cv>0 else 100-(100/(-cv))
            for i in range(2):
                param_grid_self[i]['group'] = group+f'_{perc:.2f}'
                param_grid_self[i]['cv'] = cv
        else:
            for i in range(2):
                param_grid_self[i]['cv'] = cv
        param_grid_baseline['cv']=cv

        # first estimate the baseline of PARK and PARK+PDMOT
        main(param_grid_baseline)

        # Loop through each set of parameters for Self-training
        for type in ['PDMOT+AAP', 'PDMOT', 'AAP']: 
            for i in range(2):
                if type == 'aap':
                    param_grid_self[i]['name'] = param_grid_self[i]['name']+''_AAP'
                elif type == 'mot':
                    param_grid_self[i]['name'] = param_grid_self[i]['name'] + '_PDMOT'
                else:
                    param_grid_self[i]['name'] = param_grid_self[i]['name'].strip('_AAP').strip('_PDMOT')

            for params in param_grid_self:
                method = params['method']
                # If the method is 'threshold' or 'k_best', generate configurations accordingly
                if method == 'threshold':
                    continue
                    for thresh in params['threshold']:
                        params_copy = params.copy()
                        params_copy['threshold'] = thresh
                        for maxiter in params['max_iter']:
                            params_copy_2=params_copy.copy()
                            params_copy_2['max_iter']=maxiter
                            params_copy_2['name'] = params_copy['name'] + f'_thresh{thresh}_maxiter{maxiter}'
                            res=main(params_copy_2)
                            print(params_copy_2)
                            if res!=abs(cv): #if all the folds did not reach max_iter, it is pointless to explore bigger max_iter values
                                break

                elif method == 'k_best':
                    for k_best in params['k_best']:
                        params_copy = params.copy()
                        params_copy['k_best'] = k_best
                        for maxiter in params['max_iter']:
                            params_copy_2 = params_copy.copy()
                            params_copy_2['max_iter'] = maxiter
                            params_copy_2['name'] = params_copy['name'] + f'_kbest{k_best}_maxiter{maxiter}'
                            res = main(params_copy_2)
                            print(params_copy_2)
                            if res != abs(cv):  # if all the folds did not reach max_iter, it is pointless to explore bigger max_iter values
                                break



