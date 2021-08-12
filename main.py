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
    plt.legend(title=None, labels=['Poisonous', 'Edible'])  # TODO - maybe this is the other way oround! check!
    plt.savefig("edible_vs_poisonous3_odor_partial.png")


def set_data(data):
    df = pd.DataFrame(data)
    # set to 1 / 0 for attributes with only 2 options
    df['is-edible'] = np.where(df['is-edible'] == 'e', 1, 0)
    df['bruises'] = np.where(df['bruises'] == 't', 1, 0)
    df['gill-size'] = np.where(df['gill-size'] == 'b', 1, 0)
    df['stalk-shape'] = np.where(df['stalk-shape'] == 'e', 1, 0)
    df['veil-type'] = np.where(df['veil-type'] == 'p', 1, 0)

    # print(df.head())
    y = df['is-edible']
    x = df.drop(['is-edible', 'bruises', 'gill-size', 'stalk-shape',
                 'veil-type'],
                axis=1)
    df2 = pd.get_dummies(x, drop_first=True)
    df_final = pd.concat([df, df2], axis=1)
    df_final.drop(columns=['cap-shape', 'cap-surface', 'cap-color', 'odor',
                           'gill-color', 'stalk-surface-above-ring',
                           'stalk-surface-below-ring', 'stalk-color-above-ring',
                           'stalk-color-below-ring', 'veil-color', 'ring-type',
                           'spore-print-color', 'population', 'habitat'],
                  inplace=True)
    df_final.to_csv('mushroom_data_binary.csv')
    return df_final


# get data
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

data = pd.read_csv("mushrooms_data.csv")
# print_table(data)
# basic_plots(data)
df = set_data(data)

