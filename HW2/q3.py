import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as skl
import sklearn.metrics as skm

def _eval_perf(model, x, y):
    # your code here
    return acc, auc


def _eval_model(model, trainx, trainy, valx, valy):
    # your code here
    return {"train-acc": train_acc, "train-auc": train_auc,
            "val-acc": val_acc, "val-auc": val_auc}


def generate_train_val(x, y, valsize):
    # your code here
    # Make sure data and label stay next to each other after shuffling
    n,p = np.shape(x)
    rng = np.random.default_rng()
    xy = np.hstack((x, y.reshape(n,1)))
    rng.shuffle(xy)

    trainxy = xy[int(valsize+1):, :]
    testxy = xy[:int(valsize),:]
    train_x = trainxy[:,:-1]
    train_y = trainxy[:,-1]
    test_x = testxy[:,:-1]
    test_y = testxy[:,-1]

    #ymini = ymini.flatten()
    return {"train-x": train_x, "train-y": train_y,
            "val-x": test_x, "val-y": test_y}


def generate_kfold(x, y, k):
    # fold_assignments=np.zeros(x.shape[0])
    # your code here
    fold_assignments = np.random.randint(k, size=x.shape[0])
    return fold_assignments


def eval_holdout(x, y, valsize, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    # your code here
    genvaldict = generate_train_val(x, y, valsize)
    trainx, trainy = genvaldict['train-x'], genvaldict['train-y']
    testx, testy = genvaldict['val-x'], genvaldict['val-y']
    logistic.fit(trainx,trainy)

    y_train_pred = logistic.predict(trainx)
    y_test_pred = logistic.predict(testx)

    # Calculating Accuracy
    train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
    val_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

    # Calculating AUC
    fpr_train, tpr_train, thresholds = skm.roc_curve(trainy, y_train_pred)
    train_auc = skm.auc(fpr_train, tpr_train)
    fpr_test, tpr_test, thresholds = skm.roc_curve(testy, y_test_pred)
    val_auc = skm.auc(fpr_test, tpr_test)
    results = {"train-acc": train_acc,
               "train-auc": train_auc,
               "val-acc": val_acc,
               "val-auc": val_auc}
    return results


def eval_kfold(x, y, k, logistic):
    # generate the k-folds
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    #your code here
    # Initialize results
    train_acc_cum = np.array([])
    train_auc_cum = np.array([])
    val_acc_cum = np.array([])
    val_auc_cum = np.array([])

    # Evoke the k - Folds
    folds = generate_kfold(x, y, k)
    #Iterate through the K's
    for i in range(k):

        # Split train and test data based on K-fold assignment
        train_idx = [folds[j] !=i for j in range(len(folds))]
        test_idx = [folds[j] ==i for j in range(len(folds))]
        trainx = x[train_idx,:]
        trainy = y[train_idx]
        testx = x[test_idx,:]
        testy = y[test_idx]

        logistic.fit(trainx, trainy)

        y_train_pred = logistic.predict(trainx)
        y_test_pred = logistic.predict(testx)

        # Calculating Accuracy
        train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
        val_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

        # Calculating AUC
        fpr_train, tpr_train, thresholds = skm.roc_curve(trainy, y_train_pred)
        train_auc = skm.auc(fpr_train, tpr_train)
        fpr_test, tpr_test, thresholds = skm.roc_curve(testy, y_test_pred)
        val_auc = skm.auc(fpr_test, tpr_test)

        train_acc_cum = np.append(train_acc_cum, train_acc)
        train_auc_cum = np.append(train_auc_cum, train_auc)
        val_acc_cum = np.append(val_acc_cum, val_acc)
        val_auc_cum = np.append(val_auc_cum, val_auc)

    results = {"train-acc": np.mean(train_acc_cum),
               "train-auc": np.mean(train_auc_cum),
               "val-acc": np.mean(val_acc_cum),
               "val-auc": np.mean(val_auc_cum)}
        
    return results


def eval_mccv(x, y, valsize, s, logistic):
    results = {"train-acc": 0,
               "train-auc": 0,
               "val-acc": 0,
               "val-auc": 0}
    # your code here

    # Initialize results
    train_acc_cum = np.array([])
    train_auc_cum = np.array([])
    val_acc_cum = np.array([])
    val_auc_cum = np.array([])

    for i in range(s):
        genvaldict = generate_train_val(x, y, valsize)
        trainx, trainy = genvaldict['train-x'], genvaldict['train-y']
        testx, testy = genvaldict['val-x'], genvaldict['val-y']

        logistic.fit(trainx, trainy)

        y_train_pred = logistic.predict(trainx)
        y_test_pred = logistic.predict(testx)

        # Calculating Accuracy
        train_acc = ((trainy == y_train_pred).sum())/ trainx.shape[0]
        val_acc = ((testy == y_test_pred).sum())/ testx.shape[0]

        # Calculating AUC
        fpr_train, tpr_train, thresholds = skm.roc_curve(trainy, y_train_pred)
        train_auc = skm.auc(fpr_train, tpr_train)
        fpr_test, tpr_test, thresholds = skm.roc_curve(testy, y_test_pred)
        val_auc = skm.auc(fpr_test, tpr_test)
        
        # Append new data to the list
        train_acc_cum = np.append(train_acc_cum, train_acc)
        train_auc_cum = np.append(train_auc_cum, train_auc)
        val_acc_cum = np.append(val_acc_cum, val_acc)
        val_auc_cum = np.append(val_auc_cum, val_auc)

    results = {"train-acc": np.mean(train_acc_cum),
               "train-auc": np.mean(train_auc_cum),
               "val-acc": np.mean(val_acc_cum),
               "val-auc": np.mean(val_auc_cum)}
    
    return results
