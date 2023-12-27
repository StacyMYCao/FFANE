from sklearn.naive_bayes import MultinomialNB
import numpy as np
from scipy.stats import entropy
from sklearn.decomposition import PCA
import Levenshtein
from fasta_reader import read_fasta
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, matthews_corrcoef

import os
import pickle
import sklearn
from sklearn.svm import SVC
# from skopt.space import Real, Categorical, Integer
# from skopt import BayesSearchCV
import optuna

from lightgbm import early_stopping
from lightgbm import log_evaluation
import optuna.integration.lightgbm as lgb
import xgboost as xgb

def np2txt(matrixX,output_file):
    
    # 指定输出文件的路径
    # output_file = "output.txt"
    
    # 打开文件以写入数据
    with open(output_file, "w") as file:
        # 遍历数组列表
        for array in matrixX:
            # 使用 np.savetxt 将每个数组写入文件
            np.savetxt(file, [array], fmt="%lf", delimiter=",")
            # 添加一个空行以分隔不同的数组（可选）
            # file.write("\n")

    # 关闭文件
    file.close()
    
    print("Arrays have been written to", output_file)
def plot_roc_curves(tpr_list, fpr_list):
# def plot_roc_curves(tpr_list, fpr_list, labels,rocs):
    plt.figure(figsize=(8, 8))
    
    # 自定义线条样式和颜色
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r']
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.plot([1, 0], [0, 1], color='gray', lw=2, linestyle='--')
    
    for i in range(len(tpr_list)):
        x = fpr_list[i]
        y = tpr_list[i]
        # label = labels[i] if i < len(labels) else f'Fold {i + 1}'
        linestyle = line_styles[i % len(line_styles)]
        color = colors[i % len(colors)]
        # plt.plot(x, y, linestyle=linestyle, color=color, label= label +'=' + str(np.round(rocs[i],4)))
        plt.plot(x, y, linestyle=linestyle, color=color )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(False)
    
    plt.savefig("output_figure.png", dpi=300, bbox_inches='tight')
    plt.show()
    

def mypca(fMatrix, n_components):
    n_samples, n_features = fMatrix.shape
    n_components = min(n_samples, n_features) - 1
    pca_model = PCA(n_components=n_components)
    pca_model.fit(fMatrix)
    fMatrix2 = pca_model.fit_transform(fMatrix)
    return fMatrix2


def calculate_metrics_and_roc(y_true, y_pred,y_pred_prob):
    # Convert probability predictions to binary predictions
    # y_true = [0 if x==-1 else x for x in y_true]
    # y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc_score = matthews_corrcoef(y_true, y_pred)
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2,
    #          label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc='lower right')

    # Return computed metrics and ROC curve data
    return acc, precision, recall, f1, mcc_score, roc_auc, fpr, tpr


def show_TSNE_PN(data_pos, data_neg):
    # Concatenate the positive and negative data
    data = np.vstack((data_pos, data_neg))

    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Estimate density using KernelDensity
    # You can adjust the bandwidth as needed
    kde = KernelDensity(bandwidth=0.01)
    kde.fit(embedded_data)
    densities = np.exp(kde.score_samples(embedded_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use colormap to represent densities
    sc_pos = ax.scatter(embedded_data[:len(data_pos), 0], embedded_data[:len(data_pos), 1],
                        embedded_data[:len(data_pos), 2], c=densities[:len(data_pos)], cmap='viridis', marker='o', s=25, label='Positive')
    sc_neg = ax.scatter(embedded_data[len(data_pos):, 0], embedded_data[len(data_pos):, 1],
                        embedded_data[len(data_pos):, 2], c=densities[len(data_pos):], cmap='viridis', marker='x', s=25, label='Negative')
    
    cbar = fig.colorbar(sc_pos)
    cbar.set_label('Density')

    ax.set_title("t-SNE 3D Visualization with Density")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(loc='best')
    plt.show()
    print("Positive Data - Mean Density:", np.mean(densities[:len(data_pos)]))
    print("Positive Data - Std Density:", np.std(densities[:len(data_pos)]))
    print("Negative Data - Mean Density:", np.mean(densities[len(data_pos):]))
    print("Negative Data - Std Density:", np.std(densities[len(data_pos):]))



def show_TSNE_3D(data):
    # tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    # embedded_data = tsne.fit_transform(data)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(embedded_data[:, 0], embedded_data[:, 1], embedded_data[:, 2], marker='o', s=25)
    # ax.set_title("t-SNE 3D Visualization")
    # ax.set_xlabel("Dimension 1")
    # ax.set_ylabel("Dimension 2")
    # ax.set_zlabel("Dimension 3")
    # plt.show()
    tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=42)
    embedded_data = tsne.fit_transform(data)

    # Estimate density using KernelDensity
    # You can adjust the bandwidth as needed
    kde = KernelDensity(bandwidth=0.1)
    kde.fit(embedded_data)
    densities = np.exp(kde.score_samples(embedded_data))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use colormap to represent densities
    sc = ax.scatter(embedded_data[:, 0], embedded_data[:, 1],
                    embedded_data[:, 2], c=densities, cmap='viridis', marker='o', s=25)
    cbar = fig.colorbar(sc)
    cbar.set_label('Density')

    ax.set_title("t-SNE 3D Visualization with Density")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    plt.show()
    print(np.mean(densities))
    print(np.std(densities))


def show_TSNE_2D(X):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    X_embedded = tsne.fit_transform(X)

    # Compute kernel density estimation to estimate point densities in high-dimensional space
    kde = KernelDensity(bandwidth=0.1, kernel='gaussian')
    kde.fit(X)

    # Estimate densities for each data point
    densities = np.exp(kde.score_samples(X))

    # Plot the t-SNE embeddings colored by densities
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                c=densities, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE with Point Densities')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show()
    print(densities)


def column_z_score_normalization(data):
    # Calculate the mean and standard deviation along each column
    mean_values = np.mean(data, axis=0)
    std_values = np.std(data, axis=0)

    # Perform column-wise Z-score normalization
    normalized_data = (data - mean_values) / std_values

    return normalized_data


def column_max_min_normalization(data):
    # Calculate the maximum and minimum values along each column
    max_values = np.max(data, axis=0)
    min_values = np.min(data, axis=0)

    normalized_data = data
    # Perform column-wise max-min normalization
    for i in range(data.shape[1]):
        if (max_values[i] - min_values[i]) != 0:
            normalized_data[:, i] = (
                data[:, i] - min_values[i]) / (max_values[i] - min_values[i])

    return normalized_data

# Variance


def matrix_variance(matrix):
    variances = np.var(matrix, axis=0)  # Compute variance along columns
    normalized_variances = (variances - np.min(variances)) / \
        (np.max(variances) - np.min(variances))
    return normalized_variances

# Example usage:
# variance_scores = matrix_variance(info_matrix)
# print(variance_scores)


#Correlation:
def matrix_correlation(matrix):
    correlation_matrix = np.corrcoef(matrix, rowvar=False)
    normalized_correlation_matrix = (
        correlation_matrix + 1) / 2  # Normalize to range [0, 1]
    return normalized_correlation_matrix

# Example usage:
# correlation_scores = matrix_correlation(info_matrix)
# print(correlation_scores)


#Principal Component Analysis (PCA):
def matrix_pca(matrix, num_components=None):
    pca = PCA(n_components=num_components)
    pca.fit(matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    normalized_explained_variance_ratio = (explained_variance_ratio - np.min(
        explained_variance_ratio)) / (np.max(explained_variance_ratio) - np.min(explained_variance_ratio))
    return pca.components_, normalized_explained_variance_ratio

# Example usage:
#components, explained_variance_scores = matrix_pca(info_matrix)
#print(components)
#print(explained_variance_scores)


#Information Theory Measures:
def matrix_entropy(matrix):
    entropies = entropy(matrix.T)  # Compute entropy along columns
    normalized_entropies = 1 - (entropies / np.max(entropies))
    return normalized_entropies

# Example usage:
#entropy_scores = matrix_entropy(info_matrix)
#print(entropy_scores)
#For statistical tests such as t-tests, ANOVA, or chi-square tests, the p-value alone can be used as a score. Lower p-values generally indicate greater significance. However, please note that statistical significance may not directly reflect the effectiveness of the matrix itself.

# By normalizing the measures within the desired range, you can obtain scores that range from 0 to 1, indicating the effectiveness of the matrix based on the specific measures used.


def get_kfold_index(n, rseed, n_splits=5):
    np.random.seed(rseed)  # 设置随机种子
    indices = np.zeros(n, dtype=int)  # 存储所有折的索引信息
    # 创建KFold对象
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=rseed)
    # 根据KFold对象生成每个折的训练集和测试集索引
    i = 0
    for train_idx, test_idx in kfold.split(np.arange(n)):
        indices[test_idx] = i
        i = i+1
    return indices


def compute_Leven(Protein_List):
    Protein_List = Protein_List[:, 1]
    PPSim = np.identity(len(Protein_List))
    for i in range(len(Protein_List)):
        for j in range(i, len(Protein_List)):
            distance = Levenshtein.distance(
                Protein_List[i], Protein_List[j])    # distance指编辑距离
            p = 1 - distance/max(len(Protein_List[i]), len(Protein_List[j]))
            PPSim[i][j] = p
            PPSim[j][i] = p
    return PPSim


def GenFeatureSet(Embedding, PPI_Pos, PPI_Neg, index_pos, index_neg, cv):
    train_index_pos = PPI_Pos[np.where(index_pos != cv), :][0]
    train_index_neg = PPI_Neg[np.array(np.where(index_neg != cv)), :][0]
    test_index_pos = PPI_Pos[np.array(np.where(index_pos == cv)), :][0]
    test_index_neg = PPI_Neg[np.array(np.where(index_neg == cv)), :][0]

    TSN_Pos = np.hstack((Embedding[train_index_pos[:, 0], :],
                         Embedding[train_index_pos[:, 1], :]
                         ))

    TSN_Neg = np.hstack((Embedding[train_index_neg[:, 0], :],
                         Embedding[train_index_neg[:, 1], :]
                         ))
    TST_Pos = np.hstack((Embedding[test_index_pos[:, 0], :],
                         Embedding[test_index_pos[:, 1], :]
                         ))
    TST_Neg = np.hstack((Embedding[test_index_neg[:, 0], :],
                         Embedding[test_index_neg[:, 1], :]
                         ))

    Train_Feature = np.vstack((TSN_Pos, TSN_Neg))
    Test_Feature = np.vstack((TST_Pos, TST_Neg))

    Train_labels = np.hstack((np.ones(len(TSN_Pos), dtype=(int)),
                              np.zeros(len(TSN_Neg), dtype=(int))
                              ))
    Test_labels = np.hstack((np.ones(len(TST_Pos), dtype=(int)),
                             np.zeros(len(TST_Neg), dtype=(int))
                             ))
    return Train_Feature, Test_Feature, Train_labels, Test_labels

# def clf_svm(Train_Feature, Test_Feature, Train_labels, Test_labels, best_svc_c):
#     classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
#     classifier_obj.fit(Train_Feature, Train_labels)  # Train the classifier with the chosen hyperparameters
#     predictions = classifier_obj.predict(Test_Feature)
#     accuracy = sklearn.metrics.accuracy_score(Test_labels, predictions)
#     return accuracy


def NormX(Mx):
    num_raw, _ = Mx.shape
    for i in range(num_raw):
        sum_raw = np.sum(Mx[i, :])
        if sum_raw == 0:
            continue
        Mx[i, :] = Mx[i, :]/sum_raw
    return Mx


def MaxMinM(Mx):
    MMax = np.max(Mx)
    MMin = np.min(Mx)
    Mx = (Mx - MMin)/(MMax-MMin)
    return Mx


def GetParameterMatrix():
    alp_list = np.linspace(0.5, 1, 5)
    beta_list = np.linspace(0.7, 1, 4)
    # t_list = np.linspace(1, 7, 4).astype(int)
    t_list = [1]

    ParameterMatrix = [[alp, beta, t]
                       for alp in alp_list for beta in beta_list for t in t_list]

    return ParameterMatrix


def GetProteinInfo(firDir):
    Protein_Header = []
    Protein_Sequence = []
    for item in read_fasta(firDir):
        Protein_Header.append(item.defline)
        Protein_Sequence.append(item.sequence)
    return np.stack((Protein_Header, Protein_Sequence), axis=1)


def gaussian_similarity_matrix(data_matrix, gamma=1.0):
    n = data_matrix.shape[0]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = np.exp(-gamma *
                                             np.linalg.norm(data_matrix[i] - data_matrix[j])**2)
    return similarity_matrix


def gaussian_similarity(matrix):
    num_rows, num_cols = matrix.shape

    Gaussian_similarities = np.zeros((num_rows, num_rows))
    pare_a = 0
    sum_a = 0

    for i in range(num_rows):
        temp = np.linalg.norm(matrix[i, :])
        sum_a += temp**2

    pare_a = 1 / (sum_a / num_rows)

    for i in range(num_rows):
        for j in range(num_rows):
            diff = matrix[i, :] - matrix[j, :]
            Gaussian_similarities[i,
                                  j] = np.exp(-pare_a * np.linalg.norm(diff)**2)
    return Gaussian_similarities
    # DS = Gaussian_similarities

    # Gaussian_similarities = np.zeros((num_cols, num_cols))
    # pare_b = 0
    # sum_b = 0

    # for i in range(num_cols):
    #     temp = np.linalg.norm(matrix[:, i])
    #     sum_b += temp**2

    # pare_b = 1 / (sum_b / num_cols)

    # for i in range(num_cols):
    #     for j in range(num_cols):
    #         diff = matrix[:, i] - matrix[:, j]
    #         Gaussian_similarities[i, j] = np.exp(-pare_b * np.linalg.norm(diff)**2)

    # RS = Gaussian_similarities

    # return Gaussian_similarities


# def CLFParallel(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile):

#     if os.path.exists(tarFile):
#         pickfile = open(tarFile, 'rb')
#         opt = pickle.load(pickfile, allow_pickle=True)
#         pickfile.close()
#         ss = opt.score(Test_Feature, Test_labels)
#         return ss

#     print('opt')
#     opt = BayesSearchCV( SVC(), {
#         'C': Real(1e-10, 1e+10, prior='log-uniform'),
#         'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
#         'degree': Integer(1, 5),
#         'kernel': Categorical(['rbf'])},
#         n_iter=50, n_points=4, n_jobs=2, cv=10,
#         random_state=0, )
#     print('opt')
#     _ = opt.fit(Train_Feature, Train_labels)
#     ss = opt.score(Test_Feature, Test_labels)
#     print(ss)
#     pickfile = open(tarFile, 'wb')
#     pickle.dump(opt, pickfile)
#     pickfile.close()
#     return ss

class Objective(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels, ):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        svc_g = trial.suggest_float("svc_g", 1e-10, 1, log=True)
        degree = trial.suggest_int("degree", 1, 5)
        classifier_obj = SVC(kernel="rbf", C=svc_c, gamma=svc_g, degree=degree)
        classifier_obj.fit(self.Train_Feature, self.Train_labels)
        accuracy = classifier_obj.score(self.Test_Feature, self.Test_labels)
        return accuracy

def trainSVM(best_params,Train_Feature, Test_Feature, Train_labels, Test_labels):
    classifier_obj = SVC(kernel="rbf", C=best_params["svc_c"],
                         gamma=best_params["svc_g"],
                         degree=best_params["degree"],
                         probability=True)
    classifier_obj.fit(Train_Feature, Train_labels)
    y_pred=classifier_obj.predict(Test_Feature)
    y_probs = classifier_obj.predict_proba(Test_Feature)[:,1]
    accuracy = classifier_obj.score(Test_Feature, Test_labels)
    return accuracy, y_pred, y_probs, classifier_obj
    

def go_optSVM(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarSVMpkl):

    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()
        # tarSVMpkl = SavePath + "Model_SVM_best_params.pkl"
        if os.path.exists(tarSVMpkl):
            pickfile = open(tarSVMpkl, 'rb')
            classifier_obj = pickle.load(pickfile)
            pickfile.close()
            y_probs = classifier_obj.predict_proba(Test_Feature)[:,1]
            y_pred = classifier_obj.predict(Test_Feature)
            accuracy = classifier_obj.score(Test_Feature, Test_labels)
            print('acc:', accuracy)
            return accuracy, best_params, y_pred, y_probs
        # print(best_params)
        accuracy, y_probs, classifier_obj = trainSVM(best_params,Train_Feature, Test_Feature, Train_labels, Test_labels)
        print('Train acc:', accuracy)
        pickfile = open(tarSVMpkl, 'wb')
        pickle.dump(classifier_obj, pickfile)
        pickfile.close()
        return accuracy, best_params,y_probs, y_probs

    objective = Objective(Train_Feature, Test_Feature,
                          Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=1)
    print(study.best_trial)
    accuracy, y_pred, y_probs, classifier_obj = trainSVM(study.best_params,Train_Feature, Test_Feature, Train_labels, Test_labels)


    np.save(tarFile, study.best_params)

    return study.best_value, study.best_params, y_pred, y_probs


class ObjectiveGBM(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        dtrain = lgb.Dataset(self.Train_Feature, label=self.Train_labels)
        dval = lgb.Dataset(self.Test_Feature, label=self.Test_labels)
        gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                        callbacks=[early_stopping(100), log_evaluation(100)],)
        preds = gbm.predict(self.Test_Feature)
        pred_labels = np.rint(preds)
        print('pl',pred_labels)
        accuracy = sklearn.metrics.accuracy_score(
            self.Test_labels, pred_labels)
        return accuracy


def go_optLightGBM(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl):

    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()
        # tarpkl = SavePath + "Model_SVM_best_params.pkl"
        if os.path.exists(tarpkl):
            pickfile = open(tarpkl, 'rb')
            gbm = pickle.load(pickfile)
            pickfile.close()
            # preds = gbm.predict(Test_Feature)
            y_probs = gbm.predict_proba(Test_Feature)[:, 1]
            # pred_labels = np.rint(preds)
            accuracy = gbm.score(Test_Feature, Test_labels)
            # accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)

            print('acc:', accuracy)
            return accuracy, best_params, y_probs
        print(best_params)
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "lambda_l1": best_params["lambda_l1"],
            "lambda_l2": best_params["lambda_l2"],
            "num_leaves": best_params["num_leaves"],
            "feature_fraction": best_params["feature_fraction"],
            "bagging_fraction": best_params["bagging_fraction"],
            "bagging_freq": best_params["bagging_freq"],
            "min_child_samples": best_params["min_child_samples"],
        }
        dtrain = lgb.Dataset(Train_Feature, label=Train_labels)
        dval = lgb.Dataset(Test_Feature, label=Test_labels)
        gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                        callbacks=[early_stopping(100), log_evaluation(100)],)
        # preds = gbm.predict(Test_Feature)
        y_probs = gbm.predict_proba(Test_Feature)[:, 1]
        # pred_labels = np.rint(preds)
        # accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
        accuracy = gbm.score(Test_Feature, Test_labels)

        print('acc:', accuracy)
        pickfile = open(tarpkl, 'wb')
        pickle.dump(gbm, pickfile)
        pickfile.close()
        return accuracy, best_params, y_probs

    objective = ObjectiveGBM(
        Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    print(study.best_trial)
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": study.best_params["lambda_l1"],
        "lambda_l2": study.best_params["lambda_l2"],
        "num_leaves": study.best_params["num_leaves"],
        "feature_fraction": study.best_params["feature_fraction"],
        "bagging_fraction": study.best_params["bagging_fraction"],
        "bagging_freq": study.best_params["bagging_freq"],
        "min_child_samples": study.best_params["min_child_samples"],
    }
    dtrain = lgb.Dataset(Train_Feature, label=Train_labels)
    dval = lgb.Dataset(Test_Feature, label=Test_labels)
    gbm = lgb.train(param, dtrain, valid_sets=[dtrain, dval],
                    callbacks=[early_stopping(100), log_evaluation(100)],)
    y_probs = gbm.predict_proba(Test_Feature)[:, 1]
    # preds = gbm.predict(Test_Feature)
    np.save(tarFile, study.best_params)
    return study.best_value, study.best_params, y_probs


class ObjectiveXGB(object):
    def __init__(self, Train_Feature, Test_Feature, Train_labels, Test_labels):
        self.Train_Feature = Train_Feature
        self.Test_Feature = Test_Feature
        self.Train_labels = Train_labels
        self.Test_labels = Test_labels

    def __call__(self, trial):
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = trial.suggest_int(
                "min_child_weight", 2, 10)
            param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
            # defines how selective algorithm is.
            param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            param["grow_policy"] = trial.suggest_categorical(
                "grow_policy", ["depthwise", "lossguide"])

        if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical(
                "sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical(
                "normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_float(
                "rate_drop", 1e-8, 1.0, log=True)
            param["skip_drop"] = trial.suggest_float(
                "skip_drop", 1e-8, 1.0, log=True)

        dtrain = xgb.DMatrix(self.Train_Feature, label=self.Train_labels)
        dvalid = xgb.DMatrix(self.Test_Feature, label=self.Test_labels)

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        print('pl',pred_labels)
        accuracy = sklearn.metrics.accuracy_score(
            self.Test_labels, pred_labels)
        return accuracy





def go_optXGB(Train_Feature, Test_Feature, Train_labels, Test_labels, tarFile, tarpkl):
    dtrain = xgb.DMatrix(Train_Feature, label=Train_labels)
    dvalid = xgb.DMatrix(Test_Feature, label=Test_labels)
    if os.path.exists(tarFile):
        best_params = np.load(tarFile, allow_pickle=True)
        best_params = best_params.item()
        # tarpkl = SavePath + "Model_SVM_best_params.pkl"
        if os.path.exists(tarpkl):
            pickfile = open(tarpkl, 'rb')
            bst = pickle.load(pickfile)
            pickfile.close()
            preds = bst.predict(dvalid)
            pred_labels = np.rint(preds)
            accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)

            print('acc:', accuracy)
            return accuracy, best_params, pred_labels,preds
        print(best_params)

        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            # use exact for small dataset.
            "tree_method": "exact",
            # defines booster, gblinear for linear functions.
            "booster": best_params["booster"],
            # L2 regularization weight.
            "lambda": best_params["lambda"],
            # L1 regularization weight.
            "alpha": best_params["alpha"],
            # sampling ratio for training data.
            "subsample": best_params["subsample"],
            # sampling according to each tree.
            "colsample_bytree": best_params["colsample_bytree"],
        }
        if param["booster"] in ["gbtree", "dart"]:
            # maximum depth of the tree, signifies complexity of the tree.
            param["max_depth"] = best_params["max_depth"]
            # minimum child weight, larger the term more conservative the tree.
            param["min_child_weight"] = best_params["min_child_weight"]
            param["eta"] = best_params["eta"]
            # defines how selective algorithm is.
            param["gamma"] = best_params["gamma"]
            param["grow_policy"] = best_params["grow_policy"]

        if param["booster"] == "dart":
            param["sample_type"] = best_params["sample_type"]
            param["normalize_type"] = best_params["normalize_type"]
            param["rate_drop"] = best_params["rate_drop"]
            param["skip_drop"] = best_params["skip_drop"]

        bst = xgb.train(param, dvalid)
        preds = bst.predict(dvalid)
        pred_labels = np.rint(preds)
        accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
        # return accuracy

        print('acc:', accuracy)
        print('pl',pred_labels)
        pickfile = open(tarpkl, 'wb')
        pickle.dump(bst, pickfile)
        pickfile.close()
        return accuracy, best_params, pred_labels, preds

    objective = ObjectiveXGB(
        Train_Feature, Test_Feature, Train_labels, Test_labels)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, n_jobs=-1)
    print(study.best_trial)
    best_params = np.save(tarFile, study.best_params)
    
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": best_params["booster"],
        # L2 regularization weight.
        "lambda": best_params["lambda"],
        # L1 regularization weight.
        "alpha": best_params["alpha"],
        # sampling ratio for training data.
        "subsample": best_params["subsample"],
        # sampling according to each tree.
        "colsample_bytree": best_params["colsample_bytree"],
    }
    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = best_params["max_depth"]
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = best_params["min_child_weight"]
        param["eta"] = best_params["eta"]
        # defines how selective algorithm is.
        param["gamma"] = best_params["gamma"]
        param["grow_policy"] = best_params["grow_policy"]

    if param["booster"] == "dart":
        param["sample_type"] = best_params["sample_type"]
        param["normalize_type"] = best_params["normalize_type"]
        param["rate_drop"] = best_params["rate_drop"]
        param["skip_drop"] = best_params["skip_drop"]
    bst = xgb.train(param, dvalid)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
    # return accuracy

    print('acc:', accuracy)
    print('pl',pred_labels)
    pickfile = open(tarpkl, 'wb')
    pickle.dump(bst, pickfile)
    pickfile.close()    
    
    return study.best_value, study.best_params, pred_labels, preds


def go_NB(Train_Feature, Test_Feature, Train_labels, Test_labels):
    clf = MultinomialNB()
    clf.fit(Train_Feature, Train_labels)
    # y_probs = clf.predict_proba(Test_Feature)
    pred_labels = clf.predict(Test_Feature)
    y_probs = clf.predict_proba(Test_Feature)[:, 1]

    accuracy = sklearn.metrics.accuracy_score(Test_labels, pred_labels)
    print('acc', accuracy)
    return accuracy, pred_labels, y_probs
