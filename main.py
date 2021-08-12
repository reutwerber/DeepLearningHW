import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd


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
    plt.legend(title=None, labels=['Poisonous', 'Edible'])
    plt.savefig("edible_vs_poisonous3_odor_partial.png")


# get data
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

data = pd.read_csv("mushrooms_data.csv")
# print_table(data)
basic_plots(data)

