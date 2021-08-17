from utils import check_veil_type_is_zero, plot_corr, plot_feature_importance, plot_confusion_matrix, basic_plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def MushroomsClustering(X):
    pca = PCA(2)
    data = pca.fit_transform(X)
    model = KMeans(n_clusters=8, init="k-means++")
    label = model.fit_predict(data)
    plt.figure(figsize=(10, 10))
    odor_labels = ["almond", "anise", "creosote", "fishy", "foul",
                   "none", "pungent", "spicy"]
    uniq = np.unique(label)
    for i in uniq:
        plt.scatter(data[label == i, 0], data[label == i, 1], label=odor_labels[i])
    plt.legend()
    plt.title("K Means Clustering")
    plt.savefig('K Means Clustering (After PCA).png', format='png', dpi=500, bbox_inches='tight')


def MushroomsDecisionTree(X, X_train, X_test, y_train, y_test):
    # Decision tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    # Feature importance
    plot_feature_importance(features_list=X.columns.values, feature_importance=dt.feature_importances_)

    y_pred_dt = dt.predict(X_test)
    print("Decision Tree Classifier report: \n", classification_report(y_test, y_pred_dt))
    print("Test Accuracy: {}%".format(round(dt.score(X_test, y_test) * 100, 2)))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_dt)
    plot_confusion_matrix(cm, "Decision Tree Confusion Matrix")


def MushroomNN(X_train, X_test, y_train, y_test):
    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(20, 68, 8), max_iter=700)
    mlp.fit(X_train, y_train)
    predictions = mlp.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(cm, "Perceptron Confusion Matrix")
    print("MLP report: \n", classification_report(y_test, predictions))
    print("Test Accuracy: {}%".format(round(mlp.score(X_test, y_test) * 100, 2)))


def MushroomRandomForest(X_train, X_test, y_train,y_test, is_missing_data=False):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("Test Accuracy: {}%".format(round(rf.score(X_test, y_test) * 100, 2)))
    y_pred_dt = rf.predict(X_test)
    print("Random Forest Classifier report: \n",
          classification_report(y_test, y_pred_dt))
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred_dt)
    if is_missing_data:
        title = "Random Forest with Missing Data Confusion Matrix"
    else:
        title = "Random Forest Data Confusion Matrix"
    plot_confusion_matrix(cm, title)


def SetData(df, change_params):
    df = df.loc[df["odor"] != "m"]  # remove musty odor
    df = df.loc[df["odor"] != "-"]  # remove unknown odor
    df = df.drop("veil-type", axis=1)
    labelencoder = LabelEncoder()

    if change_params:
        # Change parameters
        # Prepare the data
        df.drop(df.index[0])
        X = df.drop(['odor'], axis=1)
        y = df['odor']
        y = labelencoder.fit_transform(y)
        X = pd.get_dummies(X, drop_first=True)
        pca = PCA(60)
        X = pca.fit_transform(X)
    else:
        for column in df.columns:
            # df[column] = labelencoder.fit_transform(df[column])
            df[column] = np.where(df[column] != "-", labelencoder.fit_transform(df[column]), np.nan)
        df = df.fillna(df.median()) #(-200)
        X = df.drop(['odor'], axis=1)
        y = df['odor']
    return X, y

# Flags
change_params = False
is_missing_data = False

# Read file & encode
df = pd.read_csv("mushrooms_data.csv")
X, y = SetData(df, change_params)
if is_missing_data:
    df_missing = pd.read_csv("mushrooms_data_missing.csv")
    new_df = pd.concat([df, df_missing])
    X_missing, y_missing = SetData(new_df, change_params)

# Some plots to get a feel of the data
basic_plots(df)

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
if is_missing_data:
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_missing, y_missing,
                                                                random_state=42, test_size=0.2)

# Models
if is_missing_data:
    MushroomRandomForest(X_train_m, X_test_m, y_train_m, y_test_m, is_missing_data=True)
else:
    # Decision tree
    MushroomsDecisionTree(X, X_train, X_test, y_train, y_test)

    # Random Forest
    MushroomRandomForest(X_train, X_test, y_train, y_test)

    # Lower dimension using PCA and cluster using K Means
    MushroomsClustering(X)

    # Neural network
    MushroomNN(X_train, X_test, y_train, y_test)
