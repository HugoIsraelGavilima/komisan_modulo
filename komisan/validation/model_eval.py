####################################################################
"""
    ARCHIVO: MODEL-EVAL
    AUTOR: DEP- INTELIGENCIA DE LA INFOMACIÓN - ÁREA: PROYECTOS
    FECHA DE CREACIÓN: 10-10-2023
    
    Descripción: el archivo contiene funciones necesarias para la evaluacion de modelos
    de machine learning

    Outputs: -
"""
####################################################################
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt 



def logit_eval(dict_, corte = 0.5):
    
    for item_ in list(dict_.keys()):
        res_= dict_[item_]
        obs, predict = res_["obs"], res_["predict"]
        
        print("*"*50)
        print(item_)
        predicted = np.where(predict >= corte, 1, 0)
        df_train = adjusted_prediction(obs, predict, predicted)
        model_metrics(df_train)
        plot_model(df_train)
        print("*"*50)



def calc_vif(X):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

def ks_metric(data, probability, real):
    """
    Return ks metric of a data set
    Data: data frame containing probability predicted dependent variable and real dependent variable
    Probability: name of the column of probabilities
    Real: name of the column of the real dependent variable
    Event = 1
    No event =0
    """
    data = data.sort_values(by=probability, ascending=True).reset_index(drop = True) #Sort the Data Frame by Probability
    data["Index"] = data.index  #create a column of index
    data['Decile'] = pd.qcut(data["Index"], 10) # Create deciles
    grouped = data.groupby('Decile', as_index = False) #divide the Data frame in deciles
    
    cumsum = 0
    ks_list = []
    
    list_cumPercentage = []
    percentage = (grouped.count()[real]-grouped.sum()[real])/((grouped.count()[real]-grouped.sum()[real]).sum()) #calculate the percentages of non events in the deciles
    for i in percentage:
        cumsum = cumsum+i
        list_cumPercentage.append(cumsum)
        
    cumsum_no_event = pd.Series(list_cumPercentage) #acumulative sum of the percentages of the non events
    cumsum_event = (grouped.sum()[real]/grouped.sum()[real].sum()).cumsum() #acumulative sum of the percentages of the events
    ks = cumsum_no_event-cumsum_event
    
    return max(ks)
               
               
               
def adjusted_prediction(y_true, probability, estimated):
    """
    Return a Data Frame with the prediction of the dependent variable according to a low and high probability
    Eliminates the row with probability bigger than the low probability and lower than the high probability
    """
    df = pd.DataFrame({'Probability':probability,'True Dependent Variable': y_true})
    df['Adjusted Prediction'] = estimated
    return df

               
               
def model_metrics(data):
    """
    Print Accuracy, KS, Gini,and other metrics of the model
    Data must contain columns of the Adjusted y, Probability and True y
    """
    prediction = data['Adjusted Prediction'] #Take the predicted Y 
    probability = data['Probability'] #Take the probabilities f
    y_true = data['True Dependent Variable']
    
    print("Accuracy:",accuracy_score(y_true, prediction))
    print('ks:',ks_metric(data,'Probability','True Dependent Variable'))
    print("ROC AUC:", roc_auc_score(y_true,probability))
    print("Gini:", (2*roc_auc_score(y_true, probability) -1))
    tp, fn, fp, tn = confusion_matrix(y_true,prediction).ravel() #Positive=0 and Negative=1
    print('Negative Predicted value: ',(tn/(tn+fn)) )
    print('Positive Predicted value: ',(tp/(tp+fp)))
    print('Sensitivity: ',(tp/(tp+fn)) )
    print('Specificity: ',(tn/(fp+tn)) )
    print('tp, fn, fp, tn=: ',[tp, fn, fp, tn])
    print("Confusion Matrix:")
    print(confusion_matrix(y_true,prediction))
    print(classification_report(y_true,prediction))    
               
               
def plot_model(df):
    #ROC AUC curve
    ns_probs = [0 for _ in range(len(df['True Dependent Variable']))]
    lr_probs = df['Probability']
    ns_fpr, ns_tpr, _ = roc_curve(df['True Dependent Variable'], ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(df['True Dependent Variable'], lr_probs)

    #Odds Ratio

    df_decil = df.sort_values(by="Probability", ascending=True).reset_index(drop = True) #Sort the Data Frame by Probability
    df_decil["Index"]=df_decil.index  #create a column of index
    df_decil['Decile'] = pd.qcut(df_decil["Index"], 10) # Create deciles
    grouped = df_decil.groupby('Decile', as_index = False) #divide the Data frame in deciles
    odds_train = pd.DataFrame({'% of Bad Clients': grouped.sum()['True Dependent Variable']/grouped.count()['True Dependent Variable']})
    odds_train['Odds'] = odds_train['% of Bad Clients'].apply(lambda p: p/(1-p) if p<1 else 0.99/0.01)

    cumsum=0 #used in the for loop
    list_cumPercentage=[] #used in the for loop
    percentage=(grouped.count()["True Dependent Variable"]-grouped.sum()["True Dependent Variable"])/((grouped.count()["True Dependent Variable"]-grouped.sum()["True Dependent Variable"]).sum()) #calculta the percentages of non events in the deciles
    for i in percentage:
        cumsum=cumsum+i
        list_cumPercentage.append(cumsum)

    cumsum_no_event=pd.Series(list_cumPercentage) #acumulative sum of the percentages of the non events
    cumsum_event=(grouped.sum()["True Dependent Variable"]/grouped.sum()["True Dependent Variable"].sum()).cumsum()

    #GRAPHS
    fig, axs = plt.subplots(2, 2,figsize=(12,12))

    #ROC 
    axs[0, 0].plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    axs[0, 0].plot(lr_fpr, lr_tpr, marker='.', label='Model')
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0,0].set_title('ROC')
    axs[0, 0].legend()

    #Probability Histogram

    axs[0,1].hist(df[df['True Dependent Variable']==0]['Probability'],color='purple',label='Good Clients')
    axs[0,1].hist(df[df['True Dependent Variable']==1]['Probability'],color='orange', label='Bad Clients')
    axs[0, 1].set_xlabel('Probability')
    axs[0, 1].set_ylabel('Frequency')
    axs[0,1].set_title('Probability Histogram')
    axs[0,1].legend()

    #ODDS Ratio

    axs[1,0].bar(range(0,10),odds_train['Odds'])
    axs[1,0].set_xlabel('Decil')
    axs[1,0].set_ylabel('Odds')
    axs[1,0].set_title('Odds Ratio')

    #KS Chart

    axs[1,1].plot(range(0,10),cumsum_event, label= 'Good Clients',color='purple')
    axs[1,1].plot(range(0,10),cumsum_no_event, label = 'Bad Clients',color='orange')
    axs[1,1].set_xlabel('Decil')
    axs[1,1].set_ylabel('Cumulative %')
    axs[1,1].set_title('KS Chart')
    axs[1,1].legend()    
