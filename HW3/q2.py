import numpy as np
import sklearn.linear_model as skl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
import sklearn.metrics as skm
from sklearn.metrics import r2_score, root_mean_squared_error, f1_score

def eval_randomforest(trainx, trainy, testx, testy, num_trees, max_depth, min_items):
    test_prob = np.zeros(testx.shape[0])
    # your code here
    rfc = RandomForestClassifier(n_estimators=num_trees, max_depth=max_depth, min_samples_leaf=min_items)
    # Calculating Accuracy
    rfc.fit(trainx, trainy)

    y_train_pred = rfc.predict(trainx)
    y_test_pred = rfc.predict(testx)
    train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
    test_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

    # Calculating AUC
    fpr_train, tpr_train, thresholds = roc_curve(trainy, y_train_pred)
    train_auc = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = roc_curve(testy, y_test_pred)
    test_auc = auc(fpr_test, tpr_test)
    f1 = f1_score(testy,y_test_pred)

    # Calculating Probability
    test_prob = rfc.predict_proba(testx)
    #test_prob = test_prob[1,:]

    rfc_dict = {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}

    return rfc_dict

def eval_gbdt(trainx, trainy, testx, testy, num_estimators, learning_rate):
    test_prob = np.zeros(testx.shape[0])
    # your code here
    gbc = GradientBoostingClassifier(n_estimators=num_estimators, learning_rate=learning_rate)
    gbc.fit(trainx, trainy)

    y_train_pred = gbc.predict(trainx)
    y_test_pred = gbc.predict(testx)
    train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
    test_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

    # Calculating AUC
    fpr_train, tpr_train, thresholds = roc_curve(trainy, y_train_pred)
    train_auc = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = roc_curve(testy, y_test_pred)
    test_auc = auc(fpr_test, tpr_test)
    f1 = f1_score(testy,y_test_pred)

    # Calculating Probability
    test_prob = gbc.predict_proba(testx)
    #test_prob = test_prob[1,:]

    gbdt_dict = {"train-acc": train_acc, "train-auc": train_auc,
            "test-acc": test_acc, "test-auc": test_auc,
            "test-prob": test_prob}
    
    return gbdt_dict
