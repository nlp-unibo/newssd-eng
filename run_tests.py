import contextlib
from utils import pick_configuration
from models.BertMultilingual import BertMultilingual
import tensorflow as tf
import timeit
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
import argparse

SEEDS = [42, 1000, 500]


def run_random_prior(train_conf="en", test_conf="en", conf="prior"):
    """
    Runs a random labelling in the strategy defined by conf. e.g. conf = 'prior' => random majority, conf = 'uniform' => random assignement.
    """
    res_path = f"../results/results_random_{conf}_{train_conf}_{test_conf}.txt"
    x_train, y_train, x_test, y_test, x_val, y_val = pick_configuration(train_conf, test_conf)
    model = DummyClassifier(conf)
    start_tr = timeit.default_timer()
    model.fit(x_train, y_train)
    stop_tr = timeit.default_timer()
    with open(res_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(f"Runtime train: {stop_tr - start_tr}")
            print("Validation Results:")
            start_val = timeit.default_timer()
            print(classification_report(y_val, y_pred=model.predict(x_val)) + "\n")
            stop_val = timeit.default_timer()
            print(f"Runtime inference val: {stop_val - start_val}")
            print("Test results:")
            start_test = timeit.default_timer()
            print(classification_report(y_test, y_pred=model.predict(x_test)) + "\n")
            stop_test = timeit.default_timer()
            print(f"Runtime inference test: {stop_test - start_test}")


def run_tfidf_svm(train_conf="en", test_conf="en"):
    """
    Runs an instance of a model composed by a tfidf for the encoding and the SMV as a classifier.
    """
    res_path = f"../results/results_tfidf+svm_{train_conf}_{test_conf}.txt"
    x_train, y_train, x_test, y_test, x_val, y_val = pick_configuration(train_conf, test_conf)
    vect = TfidfVectorizer().fit(x_train)
    model = SVC(class_weight="balanced")
    start_tr = timeit.default_timer()
    model.fit(vect.transform(x_train), y_train)
    stop_tr = timeit.default_timer()
    with open(res_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(f"Runtime train: {stop_tr - start_tr}")
            print("Validation Results:")
            start_val = timeit.default_timer()
            print(classification_report(y_val, y_pred=model.predict(vect.transform(x_val))) + "\n")
            stop_val = timeit.default_timer()
            print(f"Runtime inference val: {stop_val - start_val}")
            print("Test results:")
            start_test = timeit.default_timer()
            print(classification_report(y_test, y_pred=model.predict(vect.transform(x_test))) + "\n")
            stop_test = timeit.default_timer()
            print(f"Runtime inference test: {stop_test - start_test}")


def run_tfidf_lr(train_conf="en", test_conf="en"):
    """
    Runs an instance of a model composed by a tfidf for the encoding and the LogisticRegression as a classifier.
    """
    res_path = f"../results/results_tfidf+lr_{train_conf}_{test_conf}.txt"
    x_train, y_train, x_test, y_test, x_val, y_val = pick_configuration(train_conf, test_conf)
    vect = TfidfVectorizer().fit(x_train)
    model = LogisticRegression(class_weight="balanced")
    start_tr = timeit.default_timer()
    model.fit(vect.transform(x_train), y_train)
    stop_tr = timeit.default_timer()
    with open(res_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(f"Runtime train: {stop_tr - start_tr}")
            print("Validation Results:")
            start_val = timeit.default_timer()
            print(classification_report(y_val, y_pred=model.predict(vect.transform(x_val))) + "\n")
            stop_val = timeit.default_timer()
            print(f"Runtime inference val: {stop_val - start_val}")
            print("Test results:")
            start_test = timeit.default_timer()
            print(classification_report(y_test, y_pred=model.predict(vect.transform(x_test))) + "\n")
            stop_test = timeit.default_timer()
            print(f"Runtime inference test: {stop_test - start_test}")


def run_sbert_lr(train_conf="en", test_conf="en"):
    """
    Runs an instance of a model composed by sentence-bert for the encoding and the LogisticRegression as a classifier.
    """
    vect = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    res_path = f"../results/results_sbert+lr_{train_conf}_{test_conf}.txt"
    x_train, y_train, x_test, y_test, x_val, y_val = pick_configuration(train_conf, test_conf)
    model = LogisticRegression(class_weight="balanced")
    start_tr = timeit.default_timer()
    model.fit(vect.encode(x_train), y_train)
    stop_tr = timeit.default_timer()
    with open(res_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(f"Runtime train: {stop_tr - start_tr}")
            print("Validation Results:")
            start_val = timeit.default_timer()
            print(classification_report(y_val, y_pred=model.predict(vect.encode(x_val))) + "\n")
            stop_val = timeit.default_timer()
            print(f"Runtime inference val: {stop_val - start_val}")
            print("Test results:")
            start_test = timeit.default_timer()
            print(classification_report(y_test, y_pred=model.predict(vect.encode(x_test))) + "\n")
            stop_test = timeit.default_timer()
            print(f"Runtime inference test: {stop_test - start_test}")


def run_conf_bert(train_conf="en", test_conf="en", seed=42, load_weights=False):
    """"
    Runs an instance of the mbert model in the request conf, if load_weights = False the model will be trained.
    """
    tf.random.set_seed(seed)
    res_path = f"../results/results_bert_{train_conf}_{test_conf}_{seed}.txt"
    x_train, y_train, x_test, y_test, x_val, y_val = pick_configuration(train_conf, test_conf)
    maxSentenceLen = int(max(map(lambda e: len(e.split(" ")), x_train)))
    model = BertMultilingual(maxSentenceLen=maxSentenceLen)
    start_tr = timeit.default_timer()
    if load_weights:
        model.load(f"weights/weights_bert_{train_conf}_{seed}.h5")
    else:
        model.fit(x_train, y_train, xval=x_val, yval=y_val, weightPath=f"weights/weights_bert_{train_conf}_{seed}.h5")
    stop_tr = timeit.default_timer()
    with open(res_path, "w") as f:
        with contextlib.redirect_stdout(f):
            print(f"Runtime train: {stop_tr - start_tr}, load_weights: {load_weights}")
            print("Validation Results:")
            start_val = timeit.default_timer()
            print(classification_report(y_val, y_pred=model.predict(x_val)) + "\n")
            stop_val = timeit.default_timer()
            print(f"Runtime inference val: {stop_val - start_val}")
            print("Test results:")
            start_test = timeit.default_timer()
            print(classification_report(y_test, y_pred=model.predict(x_test)) + "\n")
            stop_test = timeit.default_timer()
            print(f"Runtime inference test: {stop_test - start_test}")
    tf.keras.backend.clear_session()


def run_bert(train_split="en", test_split="en", load=False):
    """
    Runs the requested mbert model with all the seeds defined in SEEDS
    """
    for seed in SEEDS:
        run_conf_bert(train_split, test_split, seed, load)


def run_all():
    train_splits = ["en", "it", "en+it"]
    test_splits = ["en", "it"]
    for train_split in train_splits:
        for test_split in test_splits:
            # run_bert(train_split, test_split)
            run_sbert_lr(train_split, test_split)
            run_tfidf_lr(train_split, test_split)
            run_tfidf_svm(train_split, test_split)
            # run_random_prior(train_split, test_split, conf = "uniform")
            # run_random_prior(train_split, test_split, conf = "prior")


if __name__ == "__main__":
    # Takes the configuration from command line
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('-trl', '--train-language', help='the language to consider for train. Supported: it|en|en+it',
                        default='en', type=str)
    parser.add_argument('-tl', '--test-language', help='the language to consider for test. Supported: it|en|en+it',
                        default='en', type=str)
    parser.add_argument('-m', '--model', help='the model to use. Supported: SBERT, MBERT, SVM, LR',
                        default='en', type=str)
    args = parser.parse_args()
    pipeline = {
        "SBERT": run_sbert_lr,
        "MBERT": run_bert,
        "SVM": run_tfidf_svm,
        "LR": run_tfidf_lr
    }
    pipeline[args.model](args.train_language, args.test_language)
