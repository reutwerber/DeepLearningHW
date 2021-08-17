import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
pd.set_option('display.max_columns',100)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import silhouette_score

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import plot_confusion_matrix
import pickle
import graphviz


def print_table(data):
    print(data)
    print(data.shape)


# plot edible / poisonous
def basic_plots(data):
    plt.figure()
    fig1 = sns.countplot(x=data['is-edible'])
    plt.title('Edible vs Poisonous')
    plt.ylabel('Mushrooms')
    fig1.set(xticklabels=['Poisonous', 'Edible'])
    plt.savefig("edible_vs_poisonous1.png")

    fig, ax = plt.subplots()
    sns.countplot(x=data['odor'], hue=data['is-edible'], palette=['black', 'blue'], data=data)
    plt.ylabel('Number of Mushrooms')
    plt.legend(title=None, labels=['Poisonous', 'Edible'])
    plt.savefig("edible_vs_poisonous2_odor.png")

    new_data = data.loc[data["odor"] != "n"]
    new_data = new_data.loc[data["odor"] != "f"]
    # print(data["odor"].value_counts())
    # there is only 1 value with odor m, so we will not use it.
    new_data = new_data.loc[data["odor"] != "m"]

    fig, ax = plt.subplots()
    plt.title('Odors: Edible vs Poisonous, Partial Results')
    sns.countplot(x=new_data['odor'], hue=new_data['is-edible'], palette=['black', 'blue'], data=new_data)
    plt.ylabel('Number of Mushrooms')
    plt.legend(title=None, labels=['Poisonous', 'Edible'])  # TODO - maybe this is the other way oround! check!
    plt.savefig("edible_vs_poisonous3_odor_partial.png")


def set_data(data):
    df = pd.DataFrame(data)
    # set odor
    df['odor'] = np.where(df['odor'] == 'a', 0, df['odor'])
    df['odor'] = np.where(df['odor'] == 'l', 1, df['odor'])
    df['odor'] = np.where(df['odor'] == 'c', 2, df['odor'])
    df['odor'] = np.where(df['odor'] == 'y', 3, df['odor'])
    df['odor'] = np.where(df['odor'] == 'f', 4, df['odor'])
    df['odor'] = np.where(df['odor'] == 'm', 5, df['odor'])
    df['odor'] = np.where(df['odor'] == 'n', 6, df['odor'])
    df['odor'] = np.where(df['odor'] == 'p', 7, df['odor'])
    df['odor'] = np.where(df['odor'] == 's', 8, df['odor'])

    # set to 1 / 0 for attributes with only 2 options
    df['is-edible'] = np.where(df['is-edible'] == 'e', 1, 0)
    df['bruises'] = np.where(df['bruises'] == 't', 1, 0)
    df['gill-size'] = np.where(df['gill-size'] == 'b', 1, 0)
    df['stalk-shape'] = np.where(df['stalk-shape'] == 'e', 1, 0)
    df['veil-type'] = np.where(df['veil-type'] == 'p', 1, 0)

    # print(df.head())
    y = df['is-edible']
    x = df.drop(['is-edible', 'bruises', 'gill-size', 'stalk-shape',
                 'veil-type', 'odor'],
                axis=1)
    df2 = pd.get_dummies(x, drop_first=True)
    df_final = pd.concat([df, df2], axis=1)
    df_final.drop(columns=['cap-shape', 'cap-surface', 'cap-color',
                           'gill-attachment', 'gill-spacing',
                           'gill-color', 'stalk-surface-above-ring',
                           'stalk-surface-below-ring', 'stalk-color-above-ring',
                           'stalk-color-below-ring', 'veil-color', 'ring-number',
                           'ring-type', 'spore-print-color', 'population', 'habitat'],
                  inplace=True)
        # stalk - color - below - ring_y - 16 values - TODO - think if should be removed
        # gill-color_r - 24 values
        # cap-color_u - 16 values
        # cap-color_r - 16 values
        # cap-color_c - 25 values
        # lines in the drop are ones with less than 10 unique samples
    # df_final = df_final.drop(columns=['spore-print-color_y', 'spore-print-color_o', 'ring-type_n',
    #                                  'veil-color_w', 'veil-color_o', 'stalk-color-below-ring_o',
    #                                 'stalk-color-below-ring_c', 'stalk-color-above-ring_o',
    #                                  'stalk-color-above-ring_c', 'gill-color_y', 'gill-color_o',
    #                                  'gill-attachment_f', 'odor_m', 'cap-surface_g', 'cap-shape_c',
    #                                  'veil-type'],
    #                         axis=1)
    df_final.to_csv('mushroom_data_binary.csv')
    return df_final


# TODO - check if it is neccesarry. change fuction
def examine_best_model(model_name):
    print(model_name.best_score_)
    print(model_name.best_params_)
    print(model_name.best_estimator_)


# TODO - check if it is necessary. change function
# function to print results with test set and prediction variable
def get_results(test, pred_variable):
    return "F1:", metrics.f1_score(test, pred_variable), "Accuracy:", metrics.accuracy_score(y_test, pred_variable)


def knn_mushroom_model(x_train, y_train):
    # KNN classifier with grid search
    knn = KNeighborsClassifier()
    neighbors = [5, 9, 13]
    metric = ['manhattan', 'euclidean']
    algorithm = ['brute', 'ball_tree']
    parameters_knn = dict(n_neighbors=neighbors, metric=metric, algorithm=algorithm)

    # training KNN model, finding best params
    gridknn = GridSearchCV(knn, parameters_knn, verbose=1)
    gridknn.fit(x_train, y_train)
    examine_best_model(gridknn)
    return gridknn

def kmeans_elbow(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=3, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig("kmeans_elbow.png")


# get data
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

data = pd.read_csv("mushrooms_data.csv")
# print_table(data)
# basic_plots(data)
df = set_data(data)

# split to training and test sets
y = df['odor']
x = df.drop(['odor'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=5)
results_dict = {}
odor_labels = ["almond", "anise", "creosote", "fishy", "foul",
               "musty", "none", "pungent", "spicy"]

kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(x_train)
print(kmeans.cluster_centers_.shape)
# the result is (5, 81), 5 clusters in 81 dimentions
# Clustering - K Means
kmeans_elbow(x)
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=7, random_state=0).fit(x_train, y_train)
pred_y = kmeans.predict(x_test)
# score = silhouette_score(x, kmeans.labels_, metric='euclidean')
print(confusion_matrix(y_test, pred_y))
print(classification_report(y_test, pred_y))
print('Silhouette Score: %.3f' % score)

# Clustering - KNN
# do_knn(x_train, y_train, x_test, y_test)
