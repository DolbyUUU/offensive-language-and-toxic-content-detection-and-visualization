settings = {
    # Set the training dataset option (1 for OLID, 2 for SOLID, 3 for both OLID and SOLID)
    "dataset_option": 3,

    # Set the tweet text preprocessing option (True for preprocessing, False for skipping)
    "enable_preprocessing": False,

    # Work only when enable_preprocessing set to True
    "enable_stopwords_removal": False,

    # Set the FastText hyperparameter option (True for self-chosen values, False for default values)
    # Default hyperparameters: lr=0.1, dim=100, ws=5, epoch=5, minCount=1, minn=3, maxn=6, neg=5, 
    # wordNgrams=1, loss='ns', bucket=2000000, thread=system cores, lrUpdateRate=100, t=0.0001
    "enable_hyperparameters": False,

    # Set the FastText hyperparameter values used in the SOLID paper
    "lr_a": 0.01,
    "lr_b": 0.01,
    "lr_c": 0.09,

    "wordNgrams_a": 2,
    "wordNgrams_b": 2,
    "wordNgrams_c": 3,

    "ws_a": 5,
    "ws_b": 5,
    "ws_c": 5,

    "loss_a": "hs",
    "loss_b": "hs",
    "loss_c": "hs",

    # Paths of training datasets
    "path_train_olid": "OLID Dataset/OLIDv1.0/olid-training-v1.0.tsv",
    "path_train_solid": "SOLID Dataset/training_all.tsv",
    "path_train_solid_a": "SOLID Dataset/task_a_distant.tsv",
    "path_train_solid_b": "SOLID Dataset/task_b_distant.tsv",
    "path_train_solid_c": "SOLID Dataset/task_c_distant.tsv",

    # Paths of test datasets
    "path_test_solid_tweets_a": "SOLID Dataset/extended_test/test_a_tweets_all.tsv",
    "path_test_solid_tweets_b": "SOLID Dataset/extended_test/test_b_tweets_all.tsv",
    "path_test_solid_tweets_c": "SOLID Dataset/extended_test/test_c_tweets_all.tsv",
    "path_test_solid_labels_a": "SOLID Dataset/extended_test/test_a_labels_all.csv",
    "path_test_solid_labels_b": "SOLID Dataset/extended_test/test_b_labels_all.csv",
    "path_test_solid_labels_c": "SOLID Dataset/extended_test/test_c_labels_all.csv",

    # The number of features to include in the LIME explanation
    "num_features": 3,
}