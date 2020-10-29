from sklearn.svm import SVC
def optimize_svm(train_x, train_y):
    k=5
    KF = KFold(n_splits=k, shuffle=True, random_state=5)
    c=[1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000,100000]
    gamma = [1e-7,1e-6,1e-5,1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000]
    for i in c:
        for j in gamma:
            for train_index, test_index in KF.split(range(0, len(train_y))):
                selected_train = train_x
                traintags = train_y
                X_train, X_test = np.array(selected_train)[train_index], np.array(selected_features)[test_index]
                Y_train, Y_test = np.array(traintags)[train_index], np.array(traintags)[test_index]
                from sklearn.svm import SVC
                model = SVC(probability=True)
                auc_item= roc_auc_score(y_true, y_score[:, 1])
