import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier as gbc
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier as gbc
import lightgbm as lgb
import xgboost as xgb
from sklearn import pipeline as pl
from sklearn.metrics import roc_curve, auc
#from sklearn import GridsearchCV
import matplotlib.pyplot as plt


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    # Replace -1 in ager_type with np.nan.
    
    
    df_encoded = df.copy()
    #replace X in CAMEO_DEUG_2015 column with NaN
    df_encoded['AGER_TYP'].replace(-1 , np.nan, inplace = True)
    
    df_encoded['CAMEO_DEUG_2015'].replace('X', np.nan, inplace = True)
    
    #replace XX in CAMEO_DEU_2015 column with NaN
    df_encoded['CAMEO_DEU_2015'].replace('XX', np.nan, inplace = True)
    
    
    #replace XX in CAMEO_INTL_2015 column with NaN
    #CAMER_intl_2015 contains up to 10 distinct attributes which is too much for onehot encoding, hence could be dropped, but we will use it
    df_encoded['CAMEO_INTL_2015'].replace('XX', np.nan, inplace = True)
    print('Missing values for CAMER_DEUG, CAMEO_DEU and CAMEO_INTL done...')
    # remove selected columns and rows, ...
    #remove selected columns
    df_na_per_column = df_encoded.isnull().mean()# calculate % of nans per column
    
    df_encoded.drop(df_na_per_column[df_na_per_column > 0.34].index, axis = 1, inplace = True)
    print('Shape after droping columns is {}'.format(df_encoded.shape))
    #remove selected rows
    df_encoded.dropna(thresh= 70, inplace = True)
    print('Missing values dropped by columns then by row done...')
    #NB grob are a summarrized repitition of the fein columns, thus we drop them and keep the detailed columns
    df_encoded.drop(['LP_FAMILIE_GROB','LP_LEBENSPHASE_GROB','LP_STATUS_GROB'], axis = 1, inplace = True)
    
    #deleter EINGEFUEGT_AM: it is a date time object, and information about it si not provided
    #D19_LETZTER_KAUF_BRANCHE has no descriptions and will be droped as well
    df_encoded.drop(['ANZ_KINDER','ANZ_HH_TITEL','ANZ_TITEL','LNR','EINGEFUEGT_AM','D19_LETZTER_KAUF_BRANCHE'], axis = 1, inplace = True)
    
    print('ANZ_KINDER, ANZ_HH_TITEL and ANZ_TITEL droped because they have too many zero values ...')
    
    print('LNR','LP_FAMILIE_GROB,LP_LEBENSPHASE_GROB, LP_STATUS_GROB, EINGEFUEGT_AM and D19_LETZTER_KAUF_BRANCHE droped ...')
    
    #re-encode mixed values in PRAEGENDE_JUGENDJAHRE
    import math

    #####create the two variables
    #####decades will have six orders: 40s, 50s, 60s, 70s, 80s and 90s
    #####Movement will have two orders: mainstream and Avantgarde
    #####Avantagarde = 0 and mainstream = 1
    #variables = ["Decade", "Movement"]
    new_df = pd.DataFrame(index = df_encoded.index)

    decade_list = list()
    movement_list = list()
    
    for item in df_encoded["PRAEGENDE_JUGENDJAHRE"]:
        if math.isnan(item):
            decade_list.append(np.nan)
            movement_list.append(np.nan)
        if item == 0.0:
            decade_list.append(np.nan)
            movement_list.append(np.nan)
        if item == 1.0:
            decade_list.append(40)
            movement_list.append(0)
        if item == 2.0:
            decade_list.append(40)
            movement_list.append(1)
        if item == 3.0:
            decade_list.append(50)
            movement_list.append(1)
        if item == 4.0:
            decade_list.append(50)
            movement_list.append(0)
        if item == 5.0:
            decade_list.append(60)
            movement_list.append(1)
        if item == 6.0:
            decade_list.append(60)
            movement_list.append(0)
        if item == 7.0:
            decade_list.append(60)
            movement_list.append(0)
        if item == 8.0:
            decade_list.append(70)
            movement_list.append(1)
        if item == 9.0:
            decade_list.append(70)
            movement_list.append(0)
        if item == 10.0:
            decade_list.append(80)
            movement_list.append(1)
        if item == 11.0:
            decade_list.append(80)
            movement_list.append(0)
        if item == 12.0:
            decade_list.append(80)
            movement_list.append(1)
        if item == 13.0:
            decade_list.append(80)
            movement_list.append(0)
        if item == 14.0:
            decade_list.append(90)
            movement_list.append(1)
        if item == 15.0:
            decade_list.append(90)
            movement_list.append(0)

    new_df["Decade"] = decade_list
    new_df["Movement"] = movement_list
    df_encoded = pd.concat([df_encoded, new_df], axis = 1)
    
    print('Praegende_jugendjahre feature engineering done ..')
    #####Drop PRAEGENDE_JUGENDJAHRE
    df_encoded.drop("PRAEGENDE_JUGENDJAHRE", axis =1, inplace = True)
    
    #create the two variables for cameo intl
    #CAMEO_wealth will have five orders: 10, 20, 30, 40, 50 and 60
    #CAMEO_life_stage will have five orders too: 1, 2, 3, 4 and 5
    cameo_df = pd.DataFrame(index = df_encoded.index)
    wealth_list = list()
    life_style_list = list()

    for cameo in df_encoded["CAMEO_INTL_2015"]:
        if math.isnan(float(cameo)):
            wealth_list.append(np.nan)
            life_style_list.append(np.nan)
        if float(cameo) == 11:
            wealth_list.append(10)
            life_style_list.append(1)
        if float(cameo) == 12:
            wealth_list.append(10)
            life_style_list.append(2)
        if float(cameo) == 13:
            wealth_list.append(10)
            life_style_list.append(3)
        if float(cameo) == 14:
            wealth_list.append(10)
            life_style_list.append(4)
        if float(cameo) == 15:
            wealth_list.append(10)
            life_style_list.append(5)
        if float(cameo) == 21:
            wealth_list.append(20)
            life_style_list.append(1)
        if float(cameo) == 22:
            wealth_list.append(20)
            life_style_list.append(2)
        if float(cameo) == 23:
            wealth_list.append(20)
            life_style_list.append(3)
        if float(cameo) == 24:
            wealth_list.append(20)
            life_style_list.append(4)
        if float(cameo) == 25:
            wealth_list.append(20)
            life_style_list.append(5)
        if float(cameo) == 31:
            wealth_list.append(30)
            life_style_list.append(1)
        if float(cameo) == 32:
            wealth_list.append(30)
            life_style_list.append(2)
        if float(cameo) == 33:
            wealth_list.append(30)
            life_style_list.append(3)
        if float(cameo) == 34:
            wealth_list.append(30)
            life_style_list.append(4)
        if float(cameo) == 35:
            wealth_list.append(30)
            life_style_list.append(5)
        if float(cameo) == 41:
            wealth_list.append(40)
            life_style_list.append(1)
        if float(cameo) == 42:
            wealth_list.append(40)
            life_style_list.append(2)
        if float(cameo) == 43:
            wealth_list.append(40)
            life_style_list.append(3)
        if float(cameo) == 44:
            wealth_list.append(40)
            life_style_list.append(4)
        if float(cameo) == 45:
            wealth_list.append(40)
            life_style_list.append(5)
        if float(cameo) == 51:
            wealth_list.append(50)
            life_style_list.append(1)
        if float(cameo) == 52:
            wealth_list.append(50)
            life_style_list.append(2)
        if float(cameo) == 53:
            wealth_list.append(50)
            life_style_list.append(3)
        if float(cameo) == 54:
            wealth_list.append(50)
            life_style_list.append(4)
        if float(cameo) == 55:
            wealth_list.append(50)
            life_style_list.append(5)
    cameo_df["CAMEO_wealth"] = wealth_list
    cameo_df["CAMEO_life_stage"] = life_style_list
    df_encoded = pd.concat([df_encoded, cameo_df], axis = 1)
    print('CAMEO_INTL_2015 feature engineering done...')
    df_encoded.drop("CAMEO_INTL_2015", axis = 1, inplace = True)
    
    #asign nan to the wohnlage column values contained 7 or 8
    #the values have no order, and would latter be substituted with mean values 
    
    df_encoded["WOHNLAGE"][df_encoded["WOHNLAGE"]==7] = np.nan 
    df_encoded["WOHNLAGE"][df_encoded["WOHNLAGE"]==8] = np.nan 
    print ('numerical encoding finished...')
    
    #one hot encode OST_WEST_KZ
    #one hot encode CAMEO_DEU_2015 AND CAMEO_DEUG_2015
    df_encoded = pd.get_dummies(df_encoded,columns = ['PLZ8_BAUMAX','ARBEIT','AKT_DAT_KL','OST_WEST_KZ','CAMEO_DEU_2015','CAMEO_DEUG_2015'])
    print ('Categorical encoding of PLZ8_BAUMAX, ARBEIT, AKT_DAT_KL, OST_WEST_KZ, CAMEO_DEU_2015, CAMEO_DEUG_2015  done...')
    
    df_encoded.fillna(df_encoded.mean(), inplace=True)
    
    
    print ('Imputation finished...')
    # Return the cleaned dataframe.
    return df_encoded

def scree_pca_plot(pca):
    # Investigate the variance accounted for by each principal component.
    num_components = len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)

    vals = pca.explained_variance_ratio_
    print(vals)
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(1,1,1)
    cumvals = np.cumsum(vals)

    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')

    plt.show()
    

def clean_kaggle_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    # Replace -1 in ager_type with np.nan.
    
    df['AGER_TYP'].replace(-1 , np.nan, inplace = True)
    df_encoded = df.copy()
    #replace X in CAMEO_DEUG_2015 column with NaN
    df_encoded['CAMEO_DEUG_2015'].replace('X', np.nan, inplace = True)

    #replace XX in CAMEO_DEU_2015 column with NaN
    df_encoded['CAMEO_DEU_2015'].replace('XX', np.nan, inplace = True)


    #replace XX in CAMEO_INTL_2015 column with NaN
    #CAMER_intl_2015 contains up to 10 distinct attributes which is too much for onehot encoding, hence could be dropped, but we will use it
    df_encoded['CAMEO_INTL_2015'].replace('XX', np.nan, inplace = True)
    print('Missing values for CAMER_DEUG, CAMEO_DEU and CAMEO_INTL done...')
    # remove selected columns and rows, ...
    #remove selected columns
    df_na_per_column = df_encoded.isnull().mean()# calculate % of nans per column
    
    df_encoded.drop(df_na_per_column[df_na_per_column > 0.34].index, axis = 1, inplace = True)
    print('Shape after droping columns is {}'.format(df_encoded.shape))

    print('Missing values dropped only by columns done...')
    #NB grob are a summarrized repitition of the fein columns, thus we drop them and keep the detailed columns
    df_encoded.drop(['LP_FAMILIE_GROB','LP_LEBENSPHASE_GROB','LP_STATUS_GROB'], axis = 1, inplace = True)
    
    #deleter EINGEFUEGT_AM: it is a date time object, and information about it si not provided
    #D19_LETZTER_KAUF_BRANCHE has no descriptions and will be droped as well
    df_encoded.drop(['ANZ_KINDER','ANZ_HH_TITEL','ANZ_TITEL','LNR','EINGEFUEGT_AM','D19_LETZTER_KAUF_BRANCHE'], axis = 1, inplace = True)
    
    print('ANZ_KINDER, ANZ_HH_TITEL and ANZ_TITEL droped because they have too many zero values ...')
    
    print('LNR','LP_FAMILIE_GROB,LP_LEBENSPHASE_GROB, LP_STATUS_GROB, EINGEFUEGT_AM and D19_LETZTER_KAUF_BRANCHE droped ...')
    
    #re-encode mixed values in PRAEGENDE_JUGENDJAHRE
    import math

    #####create the two variables
    #####decades will have six orders: 40s, 50s, 60s, 70s, 80s and 90s
    #####Movement will have two orders: mainstream and Avantgarde
    #####Avantagarde = 0 and mainstream = 1
    #variables = ["Decade", "Movement"]
    new_df = pd.DataFrame(index = df_encoded.index)

    decade_list = list()
    movement_list = list()
    
    for item in df_encoded["PRAEGENDE_JUGENDJAHRE"]:
        if math.isnan(item):
            decade_list.append(np.nan)
            movement_list.append(np.nan)
        if item == 0.0:
            decade_list.append(np.nan)
            movement_list.append(np.nan)
        if item == 1.0:
            decade_list.append(40)
            movement_list.append(0)
        if item == 2.0:
            decade_list.append(40)
            movement_list.append(1)
        if item == 3.0:
            decade_list.append(50)
            movement_list.append(1)
        if item == 4.0:
            decade_list.append(50)
            movement_list.append(0)
        if item == 5.0:
            decade_list.append(60)
            movement_list.append(1)
        if item == 6.0:
            decade_list.append(60)
            movement_list.append(0)
        if item == 7.0:
            decade_list.append(60)
            movement_list.append(0)
        if item == 8.0:
            decade_list.append(70)
            movement_list.append(1)
        if item == 9.0:
            decade_list.append(70)
            movement_list.append(0)
        if item == 10.0:
            decade_list.append(80)
            movement_list.append(1)
        if item == 11.0:
            decade_list.append(80)
            movement_list.append(0)
        if item == 12.0:
            decade_list.append(80)
            movement_list.append(1)
        if item == 13.0:
            decade_list.append(80)
            movement_list.append(0)
        if item == 14.0:
            decade_list.append(90)
            movement_list.append(1)
        if item == 15.0:
            decade_list.append(90)
            movement_list.append(0)

    new_df["Decade"] = decade_list
    new_df["Movement"] = movement_list
    df_encoded = pd.concat([df_encoded, new_df], axis = 1)
    
    print('Praegende_jugendjahre feature engineering done ..')
    #####Drop PRAEGENDE_JUGENDJAHRE
    df_encoded.drop("PRAEGENDE_JUGENDJAHRE", axis =1, inplace = True)
    
    #create the two variables for cameo intl
    #CAMEO_wealth will have five orders: 10, 20, 30, 40, 50 and 60
    #CAMEO_life_stage will have five orders too: 1, 2, 3, 4 and 5
    cameo_df = pd.DataFrame(index = df_encoded.index)
    wealth_list = list()
    life_style_list = list()

    for cameo in df_encoded["CAMEO_INTL_2015"]:
        if math.isnan(float(cameo)):
            wealth_list.append(np.nan)
            life_style_list.append(np.nan)
        if float(cameo) == 11:
            wealth_list.append(10)
            life_style_list.append(1)
        if float(cameo) == 12:
            wealth_list.append(10)
            life_style_list.append(2)
        if float(cameo) == 13:
            wealth_list.append(10)
            life_style_list.append(3)
        if float(cameo) == 14:
            wealth_list.append(10)
            life_style_list.append(4)
        if float(cameo) == 15:
            wealth_list.append(10)
            life_style_list.append(5)
        if float(cameo) == 21:
            wealth_list.append(20)
            life_style_list.append(1)
        if float(cameo) == 22:
            wealth_list.append(20)
            life_style_list.append(2)
        if float(cameo) == 23:
            wealth_list.append(20)
            life_style_list.append(3)
        if float(cameo) == 24:
            wealth_list.append(20)
            life_style_list.append(4)
        if float(cameo) == 25:
            wealth_list.append(20)
            life_style_list.append(5)
        if float(cameo) == 31:
            wealth_list.append(30)
            life_style_list.append(1)
        if float(cameo) == 32:
            wealth_list.append(30)
            life_style_list.append(2)
        if float(cameo) == 33:
            wealth_list.append(30)
            life_style_list.append(3)
        if float(cameo) == 34:
            wealth_list.append(30)
            life_style_list.append(4)
        if float(cameo) == 35:
            wealth_list.append(30)
            life_style_list.append(5)
        if float(cameo) == 41:
            wealth_list.append(40)
            life_style_list.append(1)
        if float(cameo) == 42:
            wealth_list.append(40)
            life_style_list.append(2)
        if float(cameo) == 43:
            wealth_list.append(40)
            life_style_list.append(3)
        if float(cameo) == 44:
            wealth_list.append(40)
            life_style_list.append(4)
        if float(cameo) == 45:
            wealth_list.append(40)
            life_style_list.append(5)
        if float(cameo) == 51:
            wealth_list.append(50)
            life_style_list.append(1)
        if float(cameo) == 52:
            wealth_list.append(50)
            life_style_list.append(2)
        if float(cameo) == 53:
            wealth_list.append(50)
            life_style_list.append(3)
        if float(cameo) == 54:
            wealth_list.append(50)
            life_style_list.append(4)
        if float(cameo) == 55:
            wealth_list.append(50)
            life_style_list.append(5)
    cameo_df["CAMEO_wealth"] = wealth_list
    cameo_df["CAMEO_life_stage"] = life_style_list
    df_encoded = pd.concat([df_encoded, cameo_df], axis = 1)
    print('CAMEO_INTL_2015 feature engineering done...')
    df_encoded.drop("CAMEO_INTL_2015", axis = 1, inplace = True)
    
    #asign nan to the wohnlage column values contained 7 or 8
    #the values have no order, and would latter be substituted with mean values 
    
    df_encoded["WOHNLAGE"][df_encoded["WOHNLAGE"]==7] = np.nan 
    df_encoded["WOHNLAGE"][df_encoded["WOHNLAGE"]==8] = np.nan 
    print ('numerical encoding finished...')
    
    #one hot encode OST_WEST_KZ
    #one hot encode CAMEO_DEU_2015 AND CAMEO_DEUG_2015
    df_encoded = pd.get_dummies(df_encoded,columns = ['PLZ8_BAUMAX','ARBEIT','AKT_DAT_KL','OST_WEST_KZ','CAMEO_DEU_2015','CAMEO_DEUG_2015'])
    print ('Categorical encoding of PLZ8_BAUMAX, ARBEIT, AKT_DAT_KL, OST_WEST_KZ, CAMEO_DEU_2015, CAMEO_DEUG_2015  done...')
    
    df_encoded.fillna(df_encoded.mean(), inplace=True)
    
    
    print ('Imputation finished...')
    # Return the cleaned dataframe.
    return df_encoded
    
# Solution with Gradient Boost Classifier 
def GBC_test(X_train, X_test, y_train, y_test):

    gbc_pipeline = pl.make_pipeline(gbc())

    parameters = {'gradientboostingclassifier__learning_rate': [0.2,0.01,0.001,0.16,0.1],
                  'gradientboostingclassifier__n_estimators': [100,200,300,400, 500,1000,2000],
                  'gradientboostingclassifier__max_depth': [3, 5, 10,30],
                  'gradientboostingclassifier__min_samples_split': [2, 4]}

    grid_search = GridSearchCV(gbc_pipeline, parameters, scoring = 'roc_auc')

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_search.fit(X_train, y_train)

    # Get the estimator and predict
    best_clf = grid_fit.best_estimator_
    #predictions = (gbc_pipeline.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    #print("Unoptimized model\n------")
    #print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nOptimized Model\n------")
    print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))
    print("\n------")
    print(grid_fit.best_params_)

    ##ROC PLOT
    #PLOT ROC
    fpr, tpr, thresholds = roc_curve(y_test,best_predictions)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr,tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for LGBM')
    plt.legend(loc="lower right")
    plt.show()
    return best_predictions

    
    
# Solution with Light Gradient Boost Machine 
def LightGBM_test(X_train, X_test, y_train, y_test):
    
    reg = lgb.LGBMRegressor(
                learning_rate=0.001, 
                n_estimators=6000,
                lambda_l2 = 0.1
            )

    #Light GBM classifier
    clf = lgb.LGBMRegressor(random_state=0)
    parameters = {'learning_rate' : [0.01,0.001,0.16,0.1],
                  'max_depth': [3, 5, 10,30],
                  'n_estimators' : [100,200,300,400, 500,1000,2000],
                   'min_samples_split': [2, 4]
                 }

    # Perform grid search on the classifier using 'scorer' as scoring method
    grid_search = GridSearchCV(clf, parameters, scoring = 'roc_auc')

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_search.fit(X_train, y_train)

    # Get the estimator and predict
    best_clf = grid_fit.best_estimator_
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nOptimized Model\n------")
    print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))
    print("\n------")
    print(grid_fit.best_params_)

    ##ROC PLOT
    #PLOT ROC
    fpr, tpr, thresholds = roc_curve(y_test,best_predictions)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr,tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for LGBM')
    plt.legend(loc="lower right")
    plt.show()
    return best_predictions

# Solution with XGBoost  
def XGBoost_test(X_train, X_test, y_train, y_test):

    reg = lgb.LGBMRegressor(
                learning_rate=0.001, 
                n_estimators=6000,
                lambda_l2 = 0.1
            )
    #XGBOOST classifier
    clf = xgb.XGBRegressor()

    parameters = {'learning_rate' : [0.001],#[0.01,0.001,0.16,0.1],
                  'max_depth': [3],#[2,3, 5, 10,30,40],
                  'n_estimators' : [200],#[50,100,200,300,400],
                 }

    # Perform grid search on the classifier using 'scorer' as scoring method
    grid_search = GridSearchCV(clf, parameters, scoring = 'roc_auc')

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_search.fit(X_train, y_train)

    # Get the estimator and predict
    best_clf = grid_fit.best_estimator_
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, predictions)))
    print("\nOptimized Model\n------")
    print("ROC score on testing data: {:.4f}".format(roc_auc_score(y_test, best_predictions)))
    print("\n------")
    print(grid_fit.best_params_)

    ##ROC Plot
    #PLOT ROC
    fpr, tpr, thresholds = roc_curve(y_test,best_predictions)
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2

    plt.plot(fpr,tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for XGBOOST')
    plt.legend(loc="lower right")
    plt.show()
    return best_predictions, best_clf

