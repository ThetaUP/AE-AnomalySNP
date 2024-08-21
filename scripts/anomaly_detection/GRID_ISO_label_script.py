import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

def isolation_module(data):
    model_name = 'IsolationForest'

    X = data[:,:-1]
    y = data[:, -1]
    y[y==1] = -1
    y[y==0] = 1

    model_name = 'IsolationForest'

    iso_params = {'n_estimators': [100, 250, 500]}
    scoring = {"ACC": 'accuracy', "Prec": 'precision', 'Recall': 'recall'}

    iso = IsolationForest(contamination=0.5)

    clf = GridSearchCV(iso, iso_params, scoring=scoring, verbose = 1, cv=KFold(n_splits=5, shuffle=True, random_state=1), refit='ACC')

    clf.fit(X, y)
    clf_res = clf.predict(X)

    mean_test_ACC = clf.cv_results_['mean_test_ACC'][clf.best_index_]
    std_test_ACC = clf.cv_results_['std_test_ACC'][clf.best_index_]

    mean_test_Prec = clf.cv_results_['mean_test_Prec'][clf.best_index_]
    std_test_Prec = clf.cv_results_['std_test_Prec'][clf.best_index_]

    mean_test_Recall = clf.cv_results_['mean_test_Recall'][clf.best_index_]
    std_test_Recall = clf.cv_results_['std_test_Recall'][clf.best_index_]

    print(clf.best_params_)
    
    acc_full_metric = metrics.accuracy_score(y, clf_res)
    precision_full_metric, recall_full_metric, *_ =  metrics.precision_recall_fscore_support(y, clf_res, average='binary')

    return (model_name, acc_full_metric, precision_full_metric, recall_full_metric), (model_name, mean_test_ACC, std_test_ACC, mean_test_Prec, std_test_Prec, mean_test_Recall, std_test_Recall, clf.best_params_)

models_size_list = ['S', 'M', 'L']
models_PC_list = range(3, 23 + 1, 2)

df_ISO_full = pd.DataFrame(columns = ['Model',
                            'PC',
                            'Type',
                            'ACC',
                            'Prec',
                            'Recall'])

df_ISO_test = pd.DataFrame(columns = ['Model',
                            'PC',
                            'Type',
                            'ACC_test',
                            'ACC_std_test',
                            'Prec_test',
                            'Prec_std_test',
                            'Recall_test',
                            'Recall_std_test',
                            'Best_model'])

for model_size in models_size_list:
    for model_PC in models_PC_list:
        print(model_size, model_PC)

        data = pd.read_csv(f'data/Final_models_fromWCSS/Model_goodcol/Plots/MODEL_{model_size}/Errors_RAW_Reconstruction_table_Model_{model_size}_PC_{model_PC}.csv').to_numpy()

        full_metric_ISO, test_metric_ISO = isolation_module(data)
        model_name, acc_full_metric, precision_full_metric, recall_full_metric = full_metric_ISO
        model_name, mean_test_ACC, std_test_ACC, mean_test_Prec, std_test_Prec, mean_test_Recall, std_test_Recall, best_model = test_metric_ISO

        df_new_ISO_full = pd.Series({'Model': model_size,
                            'PC': model_PC,
                            'Type': model_name,
                            'ACC': acc_full_metric,
                            'Prec': precision_full_metric,
                            'Recall': recall_full_metric})
        
        df_new_ISO_test = pd.Series({'Model': model_size,
                            'PC': model_PC,
                            'Type': model_name,
                            'ACC_test': mean_test_ACC,
                            'ACC_std_test': std_test_ACC,
                            'Prec_test': mean_test_Prec,
                            'Prec_std_test': std_test_Prec,
                            'Recall_test': mean_test_Recall,
                            'Recall_std_test': std_test_Recall,
                            'Best_model': str(best_model)})

        df_ISO_full = pd.concat([df_ISO_full, df_new_ISO_full.to_frame().T], ignore_index=True)
        df_ISO_test = pd.concat([df_ISO_test, df_new_ISO_test.to_frame().T], ignore_index=True)

df_ISO_full.to_csv('data/outlier_results/GC_ISO_full_results.csv', index=False)
df_ISO_test.to_csv('data/outlier_results/GC_ISO_test_results.csv', index=False)