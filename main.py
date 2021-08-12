
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd


# get data

pd.options.display.max_columns=100
pd.options.display.max_rows=100
data = pd.read_table("mushrooms_data.txt")

print(data)
print(data.shape)