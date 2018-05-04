import pandas as pd
import numpy as np
import csv
import copy
from matplotlib import pyplot as plt
%matplotlib inline
import pandas as pd
import numpy as np
import csv
import copy
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering, DBSCAN
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, scale
from matplotlib.collections import LineCollection
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import matplotlib.cm as cm

class kmeans():
    def __init__(self, ):
        
