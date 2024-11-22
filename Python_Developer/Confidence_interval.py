rng = np.random.RandomState(seed=12345)
idx = np.arange(y_train.shape[0])

bootstrap_train_accuracies = []
bootstrap_rounds = 200
weight = 0.632

for i in range(bootstrap_rounds):

    train_idx = rng.choice(idx, size=idx.shape[0], replace=True)
    valid_idx = np.setdiff1d(idx, train_idx, assume_unique=False)

    boot_train_X, boot_train_y = X_train[train_idx], y_train[train_idx]
    boot_valid_X, boot_valid_y = X_train[valid_idx], y_train[valid_idx]

    clf.fit(boot_train_X, boot_train_y)
    valid_acc = clf.score(boot_valid_X, boot_valid_y)
    # predict training accuracy on the whole training set
    # as ib the original .632 boostrap paper
    # in Eq (6.12) in
    #    "Estimating the Error Rate of a Prediction Rule: Improvement
    #     on Cross-Validation"
    #     by B. Efron, 1983, https://doi.org/10.2307/2288636
    train_acc = clf.score(X_train, y_train)

    acc = weight * train_acc + (1.0 - weight) * valid_acc

    bootstrap_train_accuracies.append(acc)

bootstrap_train_mean = np.mean(bootstrap_train_accuracies)
bootstrap_train_mean
