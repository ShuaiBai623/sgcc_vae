import torch
import numpy as np
import os
import __init__
from utils.calculate_dist import gaussian_kl_calculation_vec, gaussian_kl_calculation_vec_pairwise, \
    calculate_mean_dist_pairwise


def kNN(train_feature, train_label, test_feature, test_label, K, sigma=0.07):
    """
    :param train_feature: N*f-dim feature (array or tensor)
    :param train_label: N array
    :param test_feature: N*f-dim feature (array or tensor)
    :param test_label: N array
    :param K: the kNN algorithm's k
    :param sigma: control sigma (always = t,0.07)
    :return:
    """
    if type(train_feature) == np.ndarray:
        train_feature = torch.from_numpy(train_feature).float()
    if type(test_feature) == np.ndarray:
        test_feature = torch.from_numpy(test_feature).float()
    if type(train_label) == np.ndarray:
        train_label = torch.from_numpy(train_label).long()
    if type(test_label) == np.ndarray:
        test_label = torch.from_numpy(test_label).long()
    train_feature = train_feature.cuda()
    test_feature = test_feature.cuda()
    train_label = train_label.cuda()
    test_label = test_label.cuda()
    C = train_label.max() + 1
    retrieval_one_hot = torch.zeros(K, C).cuda()
    dist = torch.mm(test_feature, train_feature.t())
    yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
    candidates = train_label.view(1, -1).expand(test_feature.size(0), -1)
    # n_test * N_train label , use knn
    retrieval = torch.gather(candidates, 1, yi)
    # then retrieval is a n * K tensor
    # means that [2,3,1,3] compose with the nearest cls label
    retrieval_one_hot.resize_(test_feature.size(0) * K, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
    # change from [2,3,1,3]^T into [[0,0,1,0],[0,0,0,1],[0,1,0,0],[0,0,0,1]]
    yd_transform = yd.clone().div_(sigma).exp_()
    probs = torch.sum(
        torch.mul(retrieval_one_hot.view(test_feature.size(0), -1, C), yd_transform.view(test_feature.size(0), -1, 1)),
        1)
    # then the prob result is n * C, and then we can sort
    _, predictions = probs.sort(1, True)
    # the predictions is the cls result
    # prediction : n * C means the class prob sort
    # prediction[:,0] means the most possible class we predict
    # e.g. C=10, then prediction[:,0] maybe [1,3,3,4,5,6,6,7,7,3,2,1]
    # e.g. C=10, then prediction[0,:] is the classification possible sort
    # just as [1,2,3,4,5,6,7,8,9,0], 1 is the most possible class
    # which means the most possible class we predict
    # Find which predictions match the target
    correct = predictions.eq(test_label.data.view(-1, 1))
    # eq: compare the element-wise equality
    # input : N*M other : N*1 compare each row, input : N*M other 1*M
    # compare each col
    top1 = correct.narrow(1, 0, 1).sum().item()
    # only choose the most prob
    # top5 = top5 + correct.narrow(1,0,5).sum().item()
    # choose the 5 cls the target maybe classified to
    total = test_feature.size(0)
    accuracy = top1 / total
    return accuracy


def kNN_kl_distance(k, test_mu, test_log_sigma, train_mu, train_log_sigma, test_label, train_label, GPU_flag=True,
                    weight_function="exp", **kwargs):
    """
    :param k: the k nearest neighbor
    :param test_mu: the u of test data set, numpy array, test_num*test_dim
    :param test_log_sigma: the log sigma of test data set, numpy array, test_num*test_dim
    :param train_mu: the u of train data set, numpy array, train_num*train_dim
    :param train_log_sigma: the log sigma of train data set, numpy array, train_num*train_dim
    :param test_label: the label of test data set, list or numpy array, length is test_num
    :param train_label: the label of train data set, list or numpy array, length is train_num
    :param GPU_flag: if we use GPU(undefined true)
    :param weight_function : the function to transform kl divergence into weight
    :param kwargs : some parameters for weight function,e.g. gamma
    :return: classification result
    """
    test_num, test_dim = test_mu.shape
    train_num, train_dim = train_mu.shape
    if kwargs["pre_computed"]:
        kl_dist = kwargs["kl_dist"]
        if type(kl_dist) == np.ndarray:
            kl_dist = torch.from_numpy(kl_dist).float()
    else:
        kl_dist = gaussian_kl_calculation_vec_pairwise(u1=test_mu, u2=train_mu, log_sigma1=test_log_sigma,
                                                       log_sigma2=train_log_sigma, GPU_flag=GPU_flag)
        # some little problem
        torch.cuda.empty_cache()
        if GPU_flag:
            kl_dist = torch.from_numpy(kl_dist).float().cuda()
        else:
            kl_dist = torch.from_numpy(kl_dist).float()
    kl_dist = kl_dist.cuda()
    yd, yi = kl_dist.topk(k, dim=1, largest=False, sorted=True)
    if type(train_label) == list:
        train_label = np.array(train_label)
        test_label = np.array(test_label)
    train_label = torch.from_numpy(train_label)
    test_label = torch.from_numpy(test_label)
    C = train_label.max() + 1
    candidates = train_label.view(1, -1).expand(test_num, -1).cuda()
    # test_num * train_num , use knn
    # select item as yi
    retrieval = torch.gather(candidates, 1, yi)
    retrieval_one_hot = torch.zeros(k, C).cuda()
    retrieval_one_hot.resize_(test_num * k, C).zero_()
    retrieval_one_hot.scatter_(dim=1, index=retrieval.view(-1, 1), src=torch.tensor(1))
    if weight_function == "exp":
        yd_transform = yd.clone().div_(-1 * kwargs["gamma"]).exp_().float()
    elif weight_function == "inverse":
        yd_transform = 1 / yd.clone().float()
    else:
        raise NotImplementedError("method {} not implement".format(weight_function))
    # torch.mul is an element wise broadcast multi
    probs = torch.sum(
        torch.mul(retrieval_one_hot.view(test_num, -1, C), yd_transform.view(test_num, -1, 1)),
        dim=1)
    # prob : N*C
    _, predictions = probs.sort(1, True)
    # calculate confusion matrix, and predictions is the classification result
    # for our task is a binary task
    predictions = predictions[:, 0].cpu().view(-1, 1)
    test_label = test_label.view(-1, 1)
    TP = torch.sum(predictions * test_label)
    TN = torch.sum((1 - predictions) * (1 - test_label))
    FP = torch.sum(predictions * (1 - test_label))
    FN = torch.sum((1 - predictions) * test_label)
    return yd, predictions.cpu().numpy(), int(TP), int(TN), int(FP), int(FN), kl_dist


def kNN_vae_mean(k, test_mu, train_mu, test_label, train_label, GPU_flag=True, distance="euclidean", **kwargs):
    """
    :param k: the k nearest neighbor
    :param test_mu: the u of test data set, numpy array, test_num*test_dim
    :param train_mu: the u of train data set, numpy array, train_num*train_dim
    :param test_label: the label of test data set, list or numpy array, length is test_num
    :param train_label: the label of train data set, list or numpy array, length is train_num
    :param GPU_flag: if we use GPU(undefined true)
    :param kwargs : some parameters for weight function,e.g. gamma
    :return: classification result
    """
    if kwargs["pre_computed"]:
        dist = kwargs["dist"]
        if type(dist) == np.ndarray:
            dist = torch.from_numpy(dist).float()
        if GPU_flag:
            dist = dist.cuda()
    else:
        dist = calculate_mean_dist_pairwise(u1=test_mu, u2=train_mu, GPU_flag=GPU_flag, distance=distance)
    if type(train_label) == np.ndarray:
        train_label = torch.from_numpy(train_label).long()
    if type(test_label) == np.ndarray:
        test_label = torch.from_numpy(test_label).long()
    C = train_label.max() + 1
    retrieval_one_hot = torch.zeros(k, C).cuda()
    if distance == "cosine":
        yd, yi = dist.topk(k, dim=1, largest=True, sorted=True)
        yd_transform = yd.clone().div_(kwargs["gamma"]).exp_()
    elif distance == "euclidean":
        yd, yi = dist.topk(k, dim=1, largest=False, sorted=True)
        yd_transform = yd.clone().div_(-1 * kwargs["gamma"]).exp_()
    else:
        raise NotImplementedError("distance {} not implemented".format(distance))
    if GPU_flag:
        candidates = train_label.view(1, -1).expand(test_label.size(0), -1).cuda()
    else:
        candidates = train_label.view(1, -1).expand(test_label.size(0), -1)
    # n_test * N_train label , use knn
    retrieval = torch.gather(candidates, 1, yi)
    # then retrieval is a n * K tensor
    # means that [2,3,1,3] compose with the nearest cls label
    retrieval_one_hot.resize_(test_label.size(0) * k, C).zero_()
    retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
    # change from [2,3,1,3]^T into [[0,0,1,0],[0,0,0,1],[0,1,0,0],[0,0,0,1]]
    probs = torch.sum(
        torch.mul(retrieval_one_hot.view(test_label.size(0), -1, C), yd_transform.view(test_label.size(0), -1, 1)),
        1)
    # then the prob result is n * C, and then we can sort
    _, predictions = probs.sort(1, True)
    predictions = predictions[:, 0].cpu().view(-1, 1)
    test_label = test_label.view(-1, 1)
    TP = torch.sum(predictions * test_label)
    TN = torch.sum((1 - predictions) * (1 - test_label))
    FP = torch.sum(predictions * (1 - test_label))
    FN = torch.sum((1 - predictions) * test_label)
    return yd, predictions.cpu().numpy(), int(TP), int(TN), int(FP), int(FN), dist


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # from os import path
    # import pickle
    # import re
    # from random import sample
    # from statistics import mean
    #
    # feature_path = path.join("/data/fhz/Suggest_Strategy", "Feature_Vector_Dict.pkl")
    # with open(feature_path, "rb") as f:
    #     feature_vector_dict = pickle.load(f)
    # all_img_name = feature_vector_dict["img_name_list"]
    # all_feature_vector = feature_vector_dict["feature_vector"]
    # all_img_label = np.array([int(eval(re.match("(.*)_(.*)_annotations", img_name).group(2)) > 3)
    #                           for img_name in all_img_name], dtype=np.int)
    # all_img_index = list(range(len(all_img_name)))
    # accuracy_list = list([])
    # for i in range(5):
    #     train_idx_list = sample(all_img_index, round(len(all_img_name) * 0.8))
    #     test_idx_list = [i for i in all_img_index if i not in train_idx_list]
    #     train_feature = all_feature_vector[train_idx_list, :]
    #     test_feature = all_feature_vector[test_idx_list, :]
    #     train_label = all_img_label[train_idx_list]
    #     test_label = all_img_label[test_idx_list]
    #     accuracy = kNN(train_feature, train_label, test_feature, test_label, K=50)
    #     accuracy_list.append(accuracy)
    #     print("The {}th random time ,for AutoEncoders, the accuracy is {}".format(i, accuracy))
    # print("The mean accuracy is {}".format(mean(accuracy_list)))
    # import torch
    # from os import path
    # import __init__
    # from models.densenet3d import DenseNet3d
    # from utils.dataloader import UnannotatedDataset
    # from torch.utils.data import DataLoader
    # import numpy as np
    # import re
    # import os
    # import prettytable as pt
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # # model = DenseNet().cuda().eval()
    # resume_path = "/data/fhz/unsupervised_feature_extractor_train_time_2/model_best.pth.tar"
    # checkpoint = torch.load(resume_path)
    # # model.load_state_dict(checkpoint['state_dict'])
    # # print("load network parameter successfully")
    # train_datalist = checkpoint['train_datalist']
    # test_datalist = checkpoint['test_datalist']
    # train_feature = checkpoint['train_feature']
    # test_feature = checkpoint['test_feature']
    # train_label = np.array([int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
    #                         for raw_cube_path in train_datalist], dtype=np.float)
    # test_label = np.array([int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
    #                        for raw_cube_path in test_datalist], dtype=np.float)
    # best_accuracy = 0
    # best_text = None
    # tb = pt.PrettyTable()
    # tb.field_names = ["method", "K", "gamma", "test_accuracy"]
    # for K in range(10, 61, 10):
    #     for gamma in range(8, 20):
    #         accuracy = kNN(train_feature, train_label, test_feature, test_label, K=K, sigma=1 / gamma)
    #         text = "The K is {} \n The gamma is {} \n The accuracy is {}".format(K, gamma, accuracy)
    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             best_text = text
    #         tb.add_row(["knn", K, gamma, round(accuracy, 2)])
    #     print(tb)
    # print("The best result is : ", best_text)
    1
