import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# plot edible / poisonous
def basic_plots(data):
    plt.figure(figsize=(15, 8))
    fig, ax = plt.subplots()
    sns.countplot(x=data['odor'], hue=data['is-edible'], palette=['black', 'blue'], data=data)
    plt.ylabel('Number of Mushrooms')
    plt.legend(title=None, labels=['Poisonous', 'Edible'])
    plt.title("Edible vs. Poisonous, Sorted by Odor")
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


def check_veil_type_is_zero(df):
    flag = False
    for item in df['veil-type']:
        if item != 0:
            flag = True
            print(item)
    print(flag)


def plot_corr(df):
    # Correlation
    plt.figure(figsize=(14, 12))
    sns.heatmap(df.corr(), linewidths=.1, cmap="Blues", annot=True, annot_kws={"size": 7})
    plt.yticks(rotation=0)
    plt.savefig("corr.png", format='png', dpi=400, bbox_inches='tight')


def plot_feature_importance(features_list, feature_importance):
    sorted_idx = np.argsort(feature_importance)
    plt.figure(figsize=(15, 8))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center', color="blue")
    plt.yticks(range(len(sorted_idx)), features_list[sorted_idx], fontsize=12)
    plt.xlabel('Importance')
    plt.title('Attribute importance')
    plt.savefig("featureimp.png", format='png', dpi=500, bbox_inches='tight')


def plot_confusion_matrix(cm, title):
    x_axis_labels = odor_labels
    y_axis_labels = x_axis_labels
    f, ax = plt.subplots(figsize=(7, 7))
    hm_plot = sns.heatmap(cm, annot=True, linewidths=0.2, linecolor="black", fmt=".0f", ax=ax,
                cmap="Blues", xticklabels=x_axis_labels, yticklabels=y_axis_labels)
    hm_plot.set_xticklabels(hm_plot.get_xmajorticklabels(), fontsize=9)
    hm_plot.set_yticklabels(hm_plot.get_ymajorticklabels(), fontsize=9)

    plt.xlabel("PREDICTED LABEL")
    plt.ylabel("TRUE LABEL")
    plt.title(title)
    plt.savefig(title+'.png', format='png', dpi=500, bbox_inches='tight')

odor_labels = ["almond", "anise", "creosote", "fishy", "foul",
                     "none", "pungent", "spicy"]