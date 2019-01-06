from knn import kNN_kl_distance
from os import path
from utils.inference_result import get_inference_result
from knn import kNN_kl_distance


def knn_evaluate(train_time, k, **kwargs):
    train_label, train_mu, train_log_sigma, test_label, test_mu, test_log_sigma = get_inference_result(
        train_time=train_time)
    predictions, TP, TN, FP, FN = kNN_kl_distance(k, test_mu, test_log_sigma, train_mu, train_log_sigma, test_label,
                                                  train_label, gamma=kwargs["gamma"])
    return predictions, TP, TN, FP, FN