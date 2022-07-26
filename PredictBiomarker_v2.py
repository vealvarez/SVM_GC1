# -*- coding: utf-8 -*-
"""
@author: Verónica E. Álvarez
"""

from os import path
import pandas as pandas
import numpy as np
from sklearn.model_selection import KFold 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score


def armarMatrizExistenciaUnitigsEnGenomas(path1,path2,path3):
     if path.exists(path1):
        #El archivo en path1 contiene el id y la secuencia de cada unitig
        #El archivo en path2 es una matriz de 1s y Os (existencia o no del unitig en el genoma)
        #siendo filas el ID del unitig y columna el ID del genoma
        #En el archivo de path3 quedará la matriz final donde la primera columna es la
        #secuencia del unitig y el resto de las columnas posee 1s ó 0s según el unitig esté en el genoma o no
		
		#The file in path1 contains the id and sequence of each unitig
        #The file in path2 is an array of 1 and 0 (existence or not) of the unitig in the genome
        #The rows contains the ID of the unitig and the columns contains the ID of the genome
        #The file in path3 will be the final matrix where the first column is the
        #unitig sequence and the rest of the columns have 1 or 0
       
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
    #Lee los datos de entrada y los prepara para poder procesarlos
	#Reads the input data and prepares it for processing
	
    print("Preparing data for modeling...")
    pheno = pandas.read_csv(pathMetadata,index_col=0,delimiter=';')
    pheno = pheno.dropna(subset=[phenotype]) 
    pheno = pheno[phenotype]
      
    X = pandas.read_csv(pathMatrizUnitigs, sep=" ", index_col=0, low_memory=False)
    X = X.transpose()
    X = X[X.index.isin(pheno.index)] 
    
    pheno = pheno[pheno.index.isin(X.index)]
    return X, pheno
#---------------------------------------------------------------------------------------------------
def fitmodel(X, pheno, estimator, parameters) :

    #Separa los datos en train/test sets
	#Realiza una búsqueda Grid para identificar los mejores hiperparámetros
	#Llama al predictor usando el test set con los mejores parámetros encontrados
	#Genera estadísticas sobre el predictor con los mejores parámetros encontrados
	
	#Separate data into train/test sets
	#Perform a Grid search to identify the best hyperparameters
	#Call predict on the estimator with the best found parameters using test dataset
    #Generates statistics about the predictor
	
    kfold = KFold(n_splits=5)
    nk = 0
    for train_index, test_index in kfold.split(X, pheno): 
        nk = nk + 1
        print("Running " + str(nk) + " fold...")
                
        X_train = X.iloc[train_index]
       
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]
                
        print("Performing GRID search...")
        gs_clf = GridSearchCV(estimator=estimator, param_grid=parameters, cv=5, n_jobs=-1, scoring='balanced_accuracy')
        
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
def main():
    espacioTrabajo = ""
    archUnitigs = "graph.nodes"
    archMatrizCruda = "bugwas_input.all_rows.binary"
    archMatrizUnitigs = "matrizUnitigs.txt"
    archMetadata = "metadata.csv"
    pathArchUnitigs = espacioTrabajo + archUnitigs
    pathArchMatrizUnitigs = espacioTrabajo + archMatrizUnitigs
    pathArchMatrizCruda = espacioTrabajo + archMatrizCruda
    pathMetadata = espacioTrabajo + archMetadata


    armarMatrizExistenciaUnitigsEnGenomas(pathArchUnitigs,pathArchMatrizCruda,pathArchMatrizUnitigs)
    
    phenotype="CC1"
    X, pheno = prep_data(phenotype,pathMetadata,pathArchMatrizUnitigs)
    
    svm = SVC(class_weight='balanced')
    
    svm_params = {
        'C': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1, 1.3, 1.5, 1.7, 2.0, 3.0, 4,0, 5.0, 6.0, 7.0, 8.0, 9.0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10],
        'kernel': ['linear', 'poly' , 'rbf' , 'sigmoid']
    }
   
    print("Running Support Vector Machine model ...")
    svm_model = fitmodel(X, pheno, svm, svm_params)
    
    print("Running Prediction ...")
    modeloFinal = svm.SVC(kernel="linear", gamma=0.0001, C=0.01)
    modeloFinal.fit(X,pheno)
    modeloFinal.predict(X)

    coef = modeloFinal.coef_.ravel()
    feature_names = list(X.columns)
    top_negative_coefficients = np.argsort(coef)[:100]
    print("Top negative predictors: ", np.asarray(feature_names)[top_negative_coefficients])

    top_positive_coefficients = np.argsort(coef)[-100:]
    print("Top positive predictors: ", np.asarray(feature_names)[top_positive_coefficients])

    
    
#----------------------------------------------------------------------------------------------------
    
if __name__ == "__main__":
    main()
