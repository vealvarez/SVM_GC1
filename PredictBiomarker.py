# -*- coding: utf-8 -*-
"""
@author: Dr. Verónica E. Álvarez
"""
from Bio import SeqIO
from os import path
import pandas as pandas
import itertools
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as seaborn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC 
import xgboost as xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_auc_score


def convertirArchTabAFasta(path1,path2):
# Convierte un archivo separado por tabs (que contiene en la primer columna el ID
# y en la segunda columna la secuencia) en un archivo fasta.
# Devuelve la cantidad de registros convertidos o 0 si no encontró el archivo de
# entrada.
    
# Convert a file separated by tabs (containing in the first column the ID
# and in the second column the sequence) in a file fasta.
# Returns the number of records converted or 0 if the file was not found.
# entry.
    
    if path.exists(path1):
        print("Generating unitigs fasta file...")
        count = SeqIO.convert(path1,"tab",path2,"fasta")
        return count
    else:
        print("File %s not found " %path1)
        return 0

  
#---------------------------------------------------------------------------------------------------
def armarMatrizExistenciaUnitigsEnGenomas(path1,path2,path3):
     if path.exists(path1):
        #El archivo en path1 contiene el id y la secuencia de cada unitig
        #El archivo de path2 es matriz de 1 y O (existencia o no) del unitig en el genoma
        #siendo filas el ID del unitig y columna el ID del genoma
        #En el archivo de path3 quedará la matriz final donde la primera columna es la
        #secuencia del unitig y el resto corresponde a 1 y 0 según el unitig esté en el genoma
         
        #The file in path1 contains the id and sequence of each unitig
        #The path2 file is a matrix of 1 and O (existence or not) of the unitig in the genome
        #being rows the ID of the unitig and column the ID of the genome
        #In the path3 file the final matrix will be left where the first column is the
        #unitig sequence and the rest corresponds to 1 and 0 depending on the unitig is in the genome
      
        if path.exists(path2):
            print("Generating binary matrix...")
            dataFrame1 = pandas.read_csv(path1, delimiter='\t',header=None)
            secuencias = dataFrame1.iloc[:, 1]
            dataframe2 = pandas.read_csv(path2, header=0, delimiter=" ")
            del dataframe2['ps']
            idx = 0
            new_col = secuencias     
            dataframe2.insert(loc=idx, column='Sequence', value=new_col)
            dataframe2.to_csv(path3,sep=' ',index=False)
        else:
             print("File %s not found " %path2)      
     else:
          print("File %s not found " %path1) 

#---------------------------------------------------------------------------------------------------        
def prep_data(phenotype,pathMetadata,pathMatrizUnitigs) :
    
    print("Preparing data for modeling...")
    
    pheno = pandas.read_csv(pathMetadata,index_col=0,delimiter=';')
    pheno = pheno.dropna(subset=[phenotype]) # drop samples that don't have a value for our chosen resistance profile
    pheno = pheno[phenotype]
        
   
    X = pandas.read_csv(pathMatrizUnitigs, sep=" ", index_col=0, low_memory=False)
    X = X.transpose()
    X = X[X.index.isin(pheno.index)] # only keep rows with a resistance measure
    
    pheno = pheno[pheno.index.isin(X.index)]
    return X, pheno
#---------------------------------------------------------------------------------------------------
def fitmodel(X, pheno, estimator, parameters) :
# function for fitting a model
   
    kfold = KFold(n_splits=5)
    nk = 0
    for train_index, test_index in kfold.split(X, pheno): 
        nk = nk + 1
        print("Running " + str(nk) + " fold...")
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
        
        # perform grid search to identify best hyper-parameters
        print("Performing GRID search...")
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=3, n_jobs=-1, scoring='balanced_accuracy')
        
        print("Fitting model...")
        gs_clf.fit(X_train, y_train) 
        
        print("Predicting-Train...")
      
        y_pred_train = gs_clf.predict(X_train) 
        y_pred_train[y_pred_train<0.5] = 0
        y_pred_train[y_pred_train>0.5] = 1
        
        print("Confusion matrix train for the fold " + str(nk))
        print(confusion_matrix(y_train, y_pred_train))
        print("Metrics report of training for the fold " + str(nk) +": " + classification_report(y_train, y_pred_train))
        
        y_pr = gs_clf.decision_function(X_train)
        auc = roc_auc_score(y_train, y_pr)
        print('AUC: %.3f' % auc)
        
                
        print("Predicting-Test...")
       
        y_pred = gs_clf.predict(X_test) 
        y_pred[y_pred<0.5] = 0
        y_pred[y_pred>0.5] = 1

   
        print("Best hyperparameters for the fold " + str(nk))
        print(gs_clf.best_params_)
        print("Confusion matrix test for the fold " + str(nk))
        print(confusion_matrix(y_test, y_pred))
        print("Metrics report of testing for the fold " + str(nk) +": " + classification_report(y_test, y_pred))
        
        y_pr_test = gs_clf.decision_function(X_test)
        aucTest = roc_auc_score(y_test, y_pr_test)
        print('AUC: %.3f' % aucTest)
     
  
    return gs_clf
#---------------------------------------------------------------------------------------------------
def plot_coefficients(classifier, feature_names, top_features=100):
    # function for looking at SVM feature importance
    #devuelve un array con las secuencias más significativas para cada clase
         
    coef = classifier.best_estimator_.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:] #imprime los últimos 20
    top_negative_coefficients = np.argsort(coef)[:top_features] #imprime los primeros 20
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
       
    coef_top_positive = np.empty(top_features,dtype=object)
    coef_top_negative = np.empty(top_features,dtype=object)
    coef_top_coefficients = np.empty(top_features*2,dtype=object)
    
    m=0
    for n in top_positive_coefficients:
        coef_top_positive[m] = coef[n]
        m = m + 1
    
    m=0
    for n in top_negative_coefficients:
        coef_top_negative[m] = coef[n]
        m = m + 1
   
        
    coef_top_coefficients = np.hstack([coef_top_negative,coef_top_positive])
    
    # create plot
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importances (Support Vector Machine)", y=1.08)
    colors = ['crimson' if c < 0 else 'cornflowerblue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    
    feature_names_top = np.empty(top_features*2,dtype=object)
  
    j=0
    for i in top_coefficients:
        feature_names_top[j] = feature_names[i]    
        j = j + 1
    

    plt.xticks(np.arange(0, 1 + 2 * top_features), feature_names_top, rotation=60, ha='right')
    plt.show()

    return feature_names_top, coef_top_coefficients
#---------------------------------------------------------------------------------------------------
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    seaborn.set_context("talk")
    plt.figure(figsize=(7, 5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Balanced accuracy")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring="balanced_accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(color='gainsboro')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="crimson")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="cornflowerblue")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="crimson",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="cornflowerblue",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
#---------------------------------------------------------------------------------------------------
def main():
    
    espacioTrabajo ="C:/Users/Vero/Documents/Trabajo/Proyectos/MachineLearning/Abaumannii/unitigs/dbgwas/output/step1/Resultados_dbgwas_strains2/"
    archUnitigs = "graph.nodes"
    archMatrizCruda = "bugwas_input.all_rows.binary"
    archFastaUnitigs = "unitigs.fasta"
    archMatrizUnitigs = "matrizUnitigs.txt"
    archMetadata = "metadata.csv"
    pathArchUnitigs = espacioTrabajo + archUnitigs
    pathArchFastaUnitigs = espacioTrabajo + archFastaUnitigs
    pathArchMatrizUnitigs = espacioTrabajo + archMatrizUnitigs
    pathArchMatrizCruda = espacioTrabajo + archMatrizCruda
    pathMetadata = espacioTrabajo + archMetadata

    phenotype="CC1"
    X, pheno = prep_data(phenotype,pathMetadata,pathArchMatrizUnitigs)

    unitigs = X.columns 
     
    #Support vector machine #######################################################################
    svm = SVC(class_weight='balanced')
    svm_params = {
        'C': [0.01],
        'gamma': [1e-06, 1e-05],
        'kernel': ['linear']
    }
   
    print("Running Support vector machine model ...")
    svm_model = fitmodel(X, pheno, svm, svm_params)
      
    seaborn.set_context("talk")
  
    feature_names_top, top_coefficients = plot_coefficients(svm_model, X.columns)

   
    print("Top negative predictors SVN: ", feature_names_top[:100])
    print("Top negative predictors weigth SVN: ", top_coefficients[:100])
    
  
    print("Top positive predictors SVN: ", feature_names_top[-100:])
    print("Top positive predictors weigth SVN: ", top_coefficients[-100:])
    
   
#----------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()