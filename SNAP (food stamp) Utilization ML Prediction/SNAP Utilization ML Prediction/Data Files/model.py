import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report, roc_curve, roc_auc_score, RocCurveDisplay, accuracy_score
divider = '='*80

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,  MinMaxScaler

################## False Positive Rate ##########################
def false_positive_rate(y_true, y_pred):
    
    false_positives = (y_true == 0) & (y_pred == 1) # find all values where y is negative
                                                    # but we predicted positive
    false_positive_number = false_positives.sum()

    true_negatives = (y_true == 0) & (y_pred == 0)  # find all values where y is negative
                                                    # and we predicted negative
    true_negative_number = true_negatives.sum()

    # Finally, find the ratio of (FP) to (TN + FP)
    FPR = false_positive_number/(true_negative_number + false_positive_number)
    
    return FPR

############ Logistic Model Long Summary Function ##################################
def run_log(model, x_train, y_train, threshold = .5, results = False):
    
    #fitting the Model
    model.fit(x_train, y_train)
    
    #Getting predictions
    y_pred = model.predict_proba(x_train)[:,1]      # Get class predictions
    '''Cross Validation predict_proba
    y_pred = cross_val_predict(model, x_train, y_train, cv = 5, emthod = 'predict_proba')'''
    y_pred_binary = np.where(y_pred>threshold,1,0)
    
    #Accuracy
    accuracy = accuracy_score(y_train, y_pred_binary)
    '''#Cross Validation Accuracy
    accuracy = cross_val_score(model, x_train, y_train, cv=5, scoring = 'accuracy').mean()
    recall = cross_val_score(model, x_train, y_train, cv=5, scoring = 'recall').mean()
    f1 = cross_val_score(model, x_train, y_train, cv=5, scoring = 'f1').mean()
    precision = cross_val_score(model, x_train, y_train, cv=5, scoring = 'precision').mean()'''
    
    cf_matrix = confusion_matrix(y_train, y_pred_binary)  # Generate confusion matrix

    #Transform confusion matrix to dataframe
    cf_df = pd.DataFrame(cf_matrix,               
                        columns=["Predicted Non-SNAP", "Predicted SNAP"],
                        index=["True Non-SNAP", "True SNAP"]) # label rows and columns
    
    ConfusionMatrixDisplay(confusion_matrix = cf_matrix)
    
    class_report = classification_report(y_train, y_pred_binary)
    class_report_dict = classification_report(y_train, y_pred_binary, output_dict = True)
    
    #True Positive Rate, False Positive Rate
    recall = class_report_dict['1']['recall'] 
    precision = class_report_dict['1']['precision']
    f1 = class_report_dict['1']['f1-score']
    false_pos_rate = false_positive_rate(y_train, y_pred_binary)
    
    #ROC Curve
    roc_auc = roc_auc_score(y_train, y_pred)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred, pos_label = 1)
    
    if results == True:
        print( '''
        Accuracy:                  {:.2f}
        Recall/True Positive Rate: {:.2f}
        False Positive:            {:.2f}
        AUC Score:                 {:.2f}
        {}'''.format(accuracy, recall ,false_pos_rate, roc_auc, divider))
        print(f'{cf_df}\n{divider}\n{class_report}')
    return accuracy, recall, precision, f1, false_pos_rate, roc_auc, fpr,tpr

######################## Grid Search ################################################3
def linear_gridsearch(x_crossvalidation, y_crossvalidation , score, df_output= False):
    estimators = [('normalize',None),     #1. grid provides all the steps i want to include in the pipeline
                ('model',None)]

    pipe = Pipeline(estimators)           #2. instantiate the pipeline

    parameter_grid = [             #3. specify parameters for each model in the pipeline
                {'model':[LogisticRegression()],
                 'normalize': [StandardScaler(), MinMaxScaler(), None],
                 'model__penalty': ['l1','l2'],         #penalizes insignificant features
                 'model__C': [.0001,.001, .01, 1, 10, 100],
                 'model__solver':['saga']},
    
                {'model': [SVC()],
                 'normalize':  [StandardScaler(), MinMaxScaler(), None],
                  'model__C':[.0001,.001, .01, 1, 10, 100],
                'model__kernel':['linear','rbf']},
    ]
    #4. create the gridsearch with the pipeline and the parameters
    grid = GridSearchCV(pipe, parameter_grid, scoring = score, cv=5, return_train_score = True) 
           #fit the grid on the data
    fitted_grid = grid.fit(x_crossvalidation, y_crossvalidation) 
#Scoring: specifies the optimized score metric
#cv applies crpss validation, industry standard is 5
#return_train_score computes and returns the training scores to gain insights on how different parameter settings impact overfitting/underfitting. It takes a lot of computational power though
    
    #get all the iterations of the model types in a dataframe
    grid_output_df = pd.DataFrame(fitted_grid.cv_results_)
    model_summary = pd.concat([grid_output_df.iloc[:35].sort_values('mean_test_score', ascending = False).head(5), 
                                 grid_output_df.iloc[36:].sort_values('mean_test_score', ascending = False).head(5)], 
                                 axis = 0)
    model_summary['metric_eval']=score
    
    if df_output == True:
        display(grid_output_df)
    
    return model_summary

####################### Boosted Random Forest Grid Search ##################3
def boost_tree(x,y,score,df_output = False):

    xgbc = XGBClassifier(n_jobs=1)
    
    parameters = {
        'booster': ['gbtree','dart'],
        'learning_rate': [0.001, 0.1, 0.5],
        'max_depth':[1,2,3,4,5,6,7,8,9,10],
        'n_estimators': [20,50,100, 150, 180]
        }
    grid_search_gbc = GridSearchCV(xgbc, parameters,scoring = score, cv = 5,n_jobs=2,verbose=3)
    fitted_grid= grid_search_gbc.fit(x, y)
    
    grid_output_df = pd.DataFrame(fitted_grid.cv_results_) 
    if df_output == True:
        display(grid_output_df)
    
    optimized_df =grid_output_df.sort_values('mean_test_score', ascending = False).head(5)
    optimized_df['metric_eval']=score
    
    return optimized_df

################ Threshold Graphing############################################
def metric_graph(model_type, model, x,y):
    accuracy_ls, precision_ls, recall_ls, f1_ls, fpr_ls, tpr_ls, auc_ls =[], [], [], [], [], [], []
    thresh = np.linspace(.1,.9,9)
    for t in thresh:
        accuracy, recall, precision, f1, false_pos_rate, roc_auc, fpr, tpr= run_log(model, x, y, threshold = t)
        accuracy_ls.append(accuracy)
        recall_ls.append(recall)
        precision_ls.append(precision)
        auc_ls.append(roc_auc)
        f1_ls.append(f1)
        fpr_ls.append(fpr)
        tpr_ls.append(tpr)
    
    max_indx = auc_ls.index(max(auc_ls))
    
    fig, ax = plt.subplots(1,2, figsize = (15,5))
    plt.style.use('seaborn-v0_8-poster')
    plt.suptitle(f'{model_type} Threshold Evaluation', fontsize = 24)
    plt.subplot(1,2,1)
    plt.plot(thresh, recall_ls, color='salmon', label = 'Recall')
    plt.plot(thresh, precision_ls, color = 'dodgerblue', label = 'Precision')
    plt.plot(thresh, f1_ls, color= 'mediumorchid', label = 'Accuracy')
    plt.title( 'Recall, Precision, and F1', fontsize = 14)
    plt.xlabel('Threshold', fontsize = 14)
    plt.ylabel ('Percentage (%)', fontsize = 14)
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(fpr_ls[max_indx], tpr_ls[max_indx], color='darkorange', label = f'AUC: {auc_ls[max_indx]:.2f}, thresh: {thresh[max_indx]}', lw = 2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('AUC curve')
    plt.xlabel('False Positive Rate', fontsize = 14)
    plt.ylabel('True Positive Rate', fontsize = 14)
    plt.legend()
    
    #for i in range(len(fpr_ls)):
    
    plt.tight_layout()
    plt.show()
    
