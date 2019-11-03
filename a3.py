import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import time
from scipy.stats import mode
from scipy.stats import kurtosis

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.datasets.samples_generator import make_blobs

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn.neural_network import MLPClassifier

sns.set()  # for plot styling


def load_fashion_data(train_size=0):

    # Reused from Assignment 1 (rkaufholz3), copied rather than import for simplicity

    # Ref: https://machinelearningmastery.com/quick-and-dirty-data-analysis-with-pandas/
    # Ref: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    # Data source (Fashion MNIST): https://www.kaggle.com/zalando-research/fashionmnist

    # Note: Fashion MNIST data is already split between training and test sets, in separate files

    # Load training data
    train_data = pd.read_csv('./data/fashion-mnist_train.csv')
    # print train_data.isnull().any()  # no null values

    # Shuffle training data
    # Ref: https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    train_data = train_data.sample(frac=1).reset_index(drop=True)

    # Split features (X) from labels (y)
    X = train_data.drop('label', axis=1)
    y = train_data['label']

    # Check class distribution before splitting train data
    # hist = y.hist()
    # plt.show()

    # Sample data if specified (stratified sampling to maintain proportionality of classes)
    if train_size != 0:
        X_train, X_unused, y_train, y_unused = train_test_split(X, y, train_size=train_size, random_state=42,
                                                                stratify=y)
    else:
        X_train, y_train = X, y

    # Check class distribution after splitting train data
    # hist = y_train.hist()
    # plt.show()

    # Load test data
    test_data = pd.read_csv('./data/fashion-mnist_test.csv')
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    return X_train, X_test, y_train, y_test


def load_adult_data(train_size=0):

    # Reused from Assignment 2 (rkaufholz3), copied rather than import for simplicity

    # Data source (Adult): https://archive.ics.uci.edu/ml/datasets/Adult

    # Note: Adult data is provided already split between training and test sets, in separate files.  These are merged
    # here, pre-processed as a single file, then split back into new Train / Test sets.

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'label']

    # Load and merge data
    train_data = pd.read_csv('./data/adult.data', header=None, names=column_names, skipinitialspace=True)
    test_data = pd.read_csv('./data/adult.test', header=None, names=column_names, skipinitialspace=True, skiprows=[0])
    all_data = train_data.append(test_data, ignore_index=True).reset_index(drop=True)

    print("train_data shape:", train_data.shape)
    print("test_data shape:", test_data.shape)
    print("all_data shape:", all_data.shape)
    print()

    # Segregate numerical from categorical features (dropping captital-gain and capital-loss as it's 0 for most
    numerical = ['age', 'fnlwgt', 'education-num', 'hours-per-week']
    categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                   'native-country']
    labels = ['label']
    selection = numerical + categorical + labels
    select_all_data = all_data[selection]

    # Drop instances with missing / poor data
    # print "Empty / null values:\n", select_all_data.isnull().any()  # no null values
    # print select_all_data[select_all_data.sex.isin(['Male'])]
    # print select_all_data[select_all_data.age.isnull()]
    # Remove inconsistent '.' from label
    # https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
    select_all_data['label'] = select_all_data['label'].map(lambda x: x.rstrip('.'))
    # Map '<=50K' to 0, '>50K' to 1
    select_all_data['label'] = select_all_data['label'].replace('<=50K', 0)
    select_all_data['label'] = select_all_data['label'].replace('>50K', 1)
    clean_data = select_all_data.dropna(axis=0).reset_index(drop=True)

    # clean_data.groupby('label').hist()
    # plt.show()

    print('\nclean_data shape:', clean_data.shape)
    print("\nduplicate rows count:", len(all_data[all_data.duplicated(selection)]))
    # print "\nduplicate rows:", all_data[all_data.duplicated(selection)]

    # Extract features and labels
    X = clean_data.drop('label', axis=1)
    y = clean_data['label']
    print('\nX shape:', X.shape)
    print('y shape:', y.shape)

    # Perform one-hot-encoding for categorical features
    encoded_X = pd.get_dummies(X, columns=categorical)
    print("encoded_X shape:", encoded_X.shape)
    print()

    # Split into Train and Test sets

    # Check class distribution before splitting train data
    # hist = y.hist()
    # plt.show()

    # Split data into training and testing sets (shuffled and stratified)
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.25, random_state=None, stratify=y,
                                                        shuffle=True)

    # X_train_sampled, y_train_sampled = X_train, y_train

    # Check train set sizes after sampling
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # Check class distribution after splitting train data
    # hist = y_train.hist()
    # plt.show()

    if train_size != 0:
        X_train_sampled, X_train_unused, y_train_sampled, y_train_unused = train_test_split(X_train, y_train,
                                                                                            train_size=train_size,
                                                                                            random_state=42,
                                                                                            stratify=y_train,
                                                                                            shuffle=True)
    else:
        X_train_sampled, y_train_sampled = X_train, y_train

    # Check train set sizes after sampling
    print('X_train_sampled shape:', X_train_sampled.shape)
    print('y_train_sampled shape:', y_train_sampled.shape)

    # Scale numerical features
    # https://www.kdnuggets.com/2016/10/beginners-guide-neural-networks-python-scikit-learn.html/2
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-on-a-partial-part-of-a-data-set
    X_train_scaled = X_train_sampled.copy()
    X_test_scaled = X_test.copy()
    X_train_numerical = X_train_scaled[numerical]
    X_test_numerical = X_test_scaled[numerical]
    scaler = preprocessing.StandardScaler().fit(X_train_numerical)  # Fit using only Train data
    numerical_X_train = scaler.transform(X_train_numerical)
    numerical_X_test = scaler.transform(X_test_numerical)  # transform X_test with same scaler as X_train
    X_train_scaled[numerical] = numerical_X_train
    X_test_scaled[numerical] = numerical_X_test

    print("\nX_train_scaled shape:", X_train_scaled.shape)
    print("X_test_scaled shape:", X_test_scaled.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # Select important features based on correlation analysis
    # plot_correlation(X_train_scaled, y_train)
    # Features to keep: Corr. > 0.03 and Corr. < -0.05 (to start with), based on correlation plot.
    # And dropping 'sex_Female' given inverse correlation with 'sex_Male'
    # Current...
    # features_to_keep = ['marital-status_Married-civ-spouse', 'relationship_Husband', 'education-num', 'hours-per-week',
    #                     'age', 'sex_Male', 'occupation_Exec-managerial', 'occupation_Prof-specialty',
    #                     'education_Bachelors', 'education_Masters', 'education_Prof-school', 'workclass_Self-emp-inc',
    #                     'education_Doctorate', 'relationship_Wife', 'race_White', 'workclass_Federal-gov',
    #                     'workclass_Local-gov', 'native-country_United-States', 'education_9th',
    #                     'occupation_Farming-fishing', 'education_Some-college', 'education_7th-8th',
    #                     'native-country_Mexico', 'marital-status_Widowed', 'education_10th',
    #                     'occupation_Machine-op-inspct', 'marital-status_Separated', 'workclass_Private',
    #                     'workclass_?', 'occupation_?', 'occupation_Adm-clerical', 'occupation_Handlers-cleaners',
    #                     'education_11th', 'relationship_Other-relative', 'race_Black', 'marital-status_Divorced',
    #                     'education_HS-grad', 'relationship_Unmarried', 'occupation_Other-service',
    #                     'relationship_Not-in-family', 'relationship_Own-child',
    #                     'marital-status_Never-married']

    final_X_train = X_train_scaled
    final_X_test = X_test_scaled

    # plot_correlation(final_X_train, y_train_sampled)

    print("\nfinal_X_train shape:", final_X_train.shape)
    print("final_X_test shape:", final_X_test.shape)
    print("y_train shape:", y_train_sampled.shape)
    print("y_test shape:", y_test.shape)
    print()

    return final_X_train, final_X_test, y_train_sampled, y_test


def plot_confusion_matrix(y_true, y_pred, classes, title, normalize=True):

    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html

    matrix = confusion_matrix(y_true, y_pred)
    print('\nConfusion matrix:\n', matrix)
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        print('\nNormalized confusion matrix:\n', matrix)
        title = 'Normalized ' + title
    # sns.heatmap(matrix, square=False, annot=True, fmt='.0%', annot_kws={"size": 10}, cmap='Blues', cbar=False,
    #             xticklabels=classes, yticklabels=classes)
    sns.heatmap(matrix, square=False, annot=True, fmt='.0%', cmap='Blues', cbar=False,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted label', fontsize=18)
    plt.ylabel('True label', fontsize=18)
    plt.title(title, fontsize=18, y=1.03)
    plt.tight_layout()
    plt.show()


def score_clusters(clusters, y, classes, k, plot, data):

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    # https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/

    labels = np.zeros_like(clusters)
    for i in range(10):
        mask = (clusters == i)
        labels[mask] = mode(y[mask])[0]

    # Measure the accuracy of the fitted labels on X_train (clusters) against y_train (y)
    print('\nClassification report:\n', classification_report(y, labels, target_names=classes))
    score = f1_score(y, labels, average='weighted')

    # Plot confusion matrix
    if plot:
        plot_title = 'Confusion Matrix (' + data + ', k=' + str(k) + ')'
        plot_confusion_matrix(y, labels, classes, plot_title)

    return score


def get_silhouette_scores(X, clusters, num_clusters, plot, data):

    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    silhouette_avg = silhouette_score(X, clusters)
    print('\nn=', num_clusters, 'silhouette score:', silhouette_avg)

    if plot:
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, clusters)

        y_lower = 10
        for i in range(num_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / num_clusters)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        chart_title = "Silhouette Scores (" + data + ", k=" + str(num_clusters) + ")"
        plt.title(chart_title, fontsize=20)
        plt.xlabel("Silhouette coefficient values", fontsize=18)
        plt.ylabel("Cluster label", fontsize=18)

        # The vertical line for average silhouette score of all the values
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")

        plt.yticks([])  # Clear the yaxis labels / ticks
        plt.xticks([-0.2, 0, 0.2, 0.4])

        plt.show()

        return round(silhouette_avg, 2)


def plot_kmeans_results(f1, silhouette, inertia, iterations, cluster, data):

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    # fig.suptitle(data + ' Clustering Results', fontsize=20)

    axs[0, 0].plot(cluster, inertia)
    axs[0, 0].set_xlabel('Number of clusters', fontsize=18)
    axs[0, 0].set_ylabel('Inertia', fontsize=18)
    axs[0, 0].tick_params(axis='both', labelsize=14)

    axs[0, 1].plot(cluster, silhouette)
    axs[0, 1].set_xlabel('Number of clusters', fontsize=18)
    axs[0, 1].set_ylabel('Silhouette Score', fontsize=18)
    axs[0, 1].tick_params(axis='both', labelsize=14)

    axs[1, 0].plot(cluster, iterations)
    axs[1, 0].set_xlabel('Number of clusters', fontsize=18)
    axs[1, 0].set_ylabel('Iterations', fontsize=18)
    axs[1, 0].tick_params(axis='both', labelsize=14)

    axs[1, 1].plot(cluster, f1)
    axs[1, 1].set_xlabel('Number of clusters', fontsize=18)
    axs[1, 1].set_ylabel('F1 Score', fontsize=18)
    axs[1, 1].tick_params(axis='both', labelsize=14)

    plt.grid()
    plt.tight_layout()
    plt.show()


def visualize_clusters(centers, num_clusters):

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
    title = 'Fashion MNIST Clusters ' + '(k=' + str(num_clusters) + ')'
    if num_clusters == 2:
        fig, ax = plt.subplots(1, 2, figsize=(8, 2))
    elif num_clusters == 5:
        fig, ax = plt.subplots(1, 5, figsize=(8, 2))
    elif num_clusters == 15:
        fig, ax = plt.subplots(3, 5, figsize=(8, 5))
    elif num_clusters == 20:
        fig, ax = plt.subplots(4, 5, figsize=(8, 5))
    else:
        fig, ax = plt.subplots(2, int(num_clusters / 2), figsize=(8, 5))
    fig.suptitle(title, fontsize=22)
    # centers = kmeans.cluster_centers_.reshape(num_clusters, 28, 28)
    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
    plt.show()


def kmeans_clustering(X, y, classes, data, verbose, X_test):

    # cluster_range = [10]
    cluster_range = [2, 5, 10, 15, 20]
    plot = True
    plot_fashion = False
    silhouette_scores = []
    inertia_values = []
    iterations = []
    f1_scores = []
    run_times = []
    neural = False

    if not neural:

        for num_clusters in cluster_range:

            print()
            print('************* n =', num_clusters, '***************\n')

            # Generate clusters
            t0 = time.time()
            kmeans = KMeans(n_clusters=num_clusters, n_init=20, random_state=42, n_jobs=-1)
            clusters = kmeans.fit_predict(X)
            run_times.append(time.time() - t0)
            inertia_values.append(kmeans.inertia_)
            iterations.append(kmeans.n_iter_)

            if verbose:
                # print('Centers:\n', kmeans.cluster_centers_)
                print('\nInertia:', kmeans.inertia_)
                print('\nn_iter:', kmeans.n_iter_)

            # Visualize the clusters
            if plot_fashion and data == 'Fashion MNIST':
                centers = kmeans.cluster_centers_.reshape(num_clusters, 3, 1)  # 28, 28 for pre-dim-reduction
                visualize_clusters(centers, num_clusters)

            # Evaluate cluster performance
            f1_scores.append(score_clusters(clusters, y, classes, num_clusters, plot, data))

            # Silhouette scores
            silhouette_scores.append(get_silhouette_scores(X, clusters, num_clusters, plot, data))

        if plot:
            plot_kmeans_results(f1_scores, silhouette_scores, inertia_values, iterations, cluster_range, data)

        # Export results
        result_df = pd.DataFrame({'num_clusters': cluster_range,
                                  'f1_scores': f1_scores,
                                  'silhouette_scores': silhouette_scores,
                                  'inertia_values': inertia_values,
                                  'iterations': iterations,
                                  'run_times': run_times})
        result_df.to_csv('kmeans_results.csv')

    # Return clusters as new features for NN
    num_clusters = 10
    kmeans2 = KMeans(n_clusters=num_clusters, n_init=20, random_state=42, n_jobs=-1)
    clusters_train = kmeans2.fit_transform(X)
    clusters_test = kmeans2.transform(X_test)

    return clusters_train, clusters_test


def plot_em_results(f1, silhouette, bic, iterations, cluster, data):

    fig, axs = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)

    # fig.suptitle(data + ' Clustering Results', fontsize=20)

    axs[0, 0].plot(cluster, silhouette)
    axs[0, 0].set_xlabel('Number of clusters', fontsize=18)
    axs[0, 0].set_ylabel('Silhouette Score', fontsize=18)
    axs[0, 0].tick_params(axis='both', labelsize=14)

    axs[0, 1].plot(cluster, f1)
    axs[0, 1].set_xlabel('Number of clusters', fontsize=18)
    axs[0, 1].set_ylabel('F1 Score', fontsize=18)
    axs[0, 1].tick_params(axis='both', labelsize=14)

    axs[1, 0].plot(cluster, bic)
    axs[1, 0].set_xlabel('Number of clusters', fontsize=18)
    axs[1, 0].set_ylabel('BIC Score', fontsize=18)
    axs[1, 0].tick_params(axis='both', labelsize=14)

    axs[1, 1].plot(cluster, iterations)
    axs[1, 1].set_xlabel('Number of clusters', fontsize=18)
    axs[1, 1].set_ylabel('Iterations', fontsize=18)
    axs[1, 1].tick_params(axis='both', labelsize=14)

    plt.grid()
    plt.tight_layout()
    plt.show()


def em_clustering(X, y, classes, data, verbose, X_test):

    # https://learning.oreilly.com/library/view/python-data-science/9781491912126/ch05.html#in-depth-gaussian-mixture-models

    components_range = [2, 5, 10, 15, 20]
    # components_range = [10]
    plot = True
    plot_fashion = False
    silhouette_scores = []
    f1_scores = []
    covariances = []
    iterations = []
    bic_scores = []
    run_times = []
    neural = True

    if not neural:

        for num_components in components_range:

            print()
            print('************* n =', num_components, '***************\n')

            # Generate clusters
            t0 = time.time()
            gmm = GaussianMixture(n_components=num_components, covariance_type='full', n_init=10,
                                  max_iter=100, random_state=42)
            clusters = gmm.fit_predict(X)
            run_times.append(time.time() - t0)
            print('runtime: ', time.time() - t0)
            covariances.append(gmm.covariances_)
            iterations.append(gmm.n_iter_)
            bic_scores.append(gmm.bic(X))

            if verbose:
                print('Weights:\n', gmm.weights_)
                print('\nMeans:', gmm.means_)
                print('\nMeans shape:', gmm.means_.shape)
                print('\nCovariances:', gmm.covariances_)
                print('\nIterations:', gmm.n_iter_)
                print('\nBIC:', gmm.bic(X))

            # Visualize the clusters
            if plot_fashion and data == 'Fashion MNIST':
                means = gmm.means_.reshape(num_components, 10, 8)  # 28, 28 for pre-dim-reduction
                visualize_clusters(means, num_components)

            # Evaluate cluster performance
            f1_scores.append(score_clusters(clusters, y, classes, num_components, plot, data))

            # # Silhouette scores
            silhouette_scores.append(get_silhouette_scores(X, clusters, num_components, plot, data))

        if plot:
            plot_em_results(f1_scores, silhouette_scores, bic_scores, iterations, components_range, data)

        # Export results
        result_df = pd.DataFrame({'components_range': components_range,
                                  'f1_scores': f1_scores,
                                  'silhouette_scores': silhouette_scores,
                                  'bic_scores': bic_scores,
                                  'iterations': iterations,
                                  'run_times': run_times})
        result_df.to_csv('em_results.csv')

    # Return clusters as new features for NN
    num_components = 10
    gmm = GaussianMixture(n_components=num_components, covariance_type='tied', n_init=10,
                          max_iter=100, random_state=42)

    print('starting...')

    gmm_model = gmm.fit(X)

    print('fit done...')

    clusters_train = gmm_model.predict_proba(X)

    print('train clustered....')

    clusters_test = gmm_model.predict_proba(X_test)

    print('test clustered...')

    return clusters_train, clusters_test


def plot_2d(comp, y):

    # I did not note where I got this reference from... TODO: find reference...
    # Similar to: https://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

    plt.scatter(comp[:, 0], comp[:, 1],
                c=y, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', 10))  # 2 for Adult, 10 for Fashion MNIST
    plt.xlabel('component 1', fontsize=18)
    plt.ylabel('component 2', fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.colorbar()
    plt.show()


def plot_3d(comp, y):

    # https://stackoverflow.com/questions/1985856/how-to-make-a-3d-scatter-plot-in-python
    fig = plt.figure()
    ax = Axes3D(fig)
    p = ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], c=y, alpha=0.5,
                   cmap=plt.cm.get_cmap('Spectral', 10))  # 2 for Adult, 10 for Fashion MNIST
    ax.set_xlabel('component 1', fontsize=18)
    ax.set_ylabel('component 2', fontsize=18)
    ax.set_zlabel('component 3', fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    fig.colorbar(p)

    plt.show()


def pca_analysis(X, y, data, plot, X_test):

    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

    if plot:
        # Project into a 2-D space for visualization
        pca = PCA(n_components=2)
        projected = pca.fit_transform(X)
        plot_2d(projected, y)

        # Project into a 3-D space for visualization
        pca = PCA(n_components=3)
        projected = pca.fit_transform(X)
        plot_3d(projected, y)

        # Analyze cumulative variance by number of components
        pca = PCA().fit(X)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components', fontsize=18)
        plt.ylabel('cumulative explained variance', fontsize=18)
        plt.tick_params(axis='both', labelsize=14)
        plt.show()

        print(np.cumsum(pca.explained_variance_ratio_))

    # Project onto an 'optimal' number of components, based on cumulative explained variance ratio plot (90%)
    pca2 = PCA(n_components=23)
    projected_train = pca2.fit_transform(X)
    projected_test = pca2.transform(X_test)
    print('\nPCA projected X_train', projected_train.shape)

    return projected_train, projected_test


def plot_kurtosis(comp):

    # https://seaborn.pydata.org/tutorial/distributions.html
    kurtosis_values = []
    for c in range(comp.shape[1]):
        kurt = round(kurtosis(comp[:, c]), 2)
        kurtosis_values.append(kurt)
        if comp.shape[1] <= 10:
            label = 'Component ' + str(c + 1) + ' (' + str(kurt) + ')'
            sns.distplot(comp[:, c], label=label, hist=False)
        else:
            sns.distplot(comp[:, c], hist=False)
    plt.legend()
    plt.show()
    print(kurtosis_values)

    return np.mean(kurtosis_values)


def ica_analysis(X, y, dataset, plot, X_test):

    if plot:
        # Project in 2D for visualization
        ica = FastICA(n_components=2)
        projected = ica.fit_transform(X)
        plot_2d(projected, y)

        # Project in 3D for visualization
        ica = FastICA(n_components=3)
        projected = ica.fit_transform(X)
        plot_3d(projected, y)

        sns.kdeplot(projected[:, 0], projected[:, 1])
        sns.rugplot(projected[:, 0], color="g")
        sns.rugplot(projected[:, 1], vertical=True)

        plt.show()

    # Project in n dimensions to compare kurtosis
    dim_range = [2, 3, 5, 7, 10, 20]
    ave_kurtosis = []
    for n in dim_range:
        ica = FastICA(n_components=n)
        projected = ica.fit_transform(X)
        print(projected.shape)
        # Plot distributions and kurtosis
        ave_kurtosis.append(plot_kurtosis(projected))

    print()
    print(ave_kurtosis)

    plt.plot(dim_range, ave_kurtosis)
    plt.xlabel('Number of components', fontsize=18)
    plt.ylabel('Average Kurtosis', fontsize=18)
    plt.tick_params(axis='both', labelsize=14)
    plt.show()

    # Project on to an 'optimal' number of components based on Kurtosis
    ica2 = FastICA(n_components=10)
    projected_train = ica2.fit_transform(X)
    projected_test = ica2.transform(X_test)
    print('\nICA projected X_train:', projected_train.shape)

    return projected_train, projected_test


def rp_analysis(X, y, dataset, plot, X_test):

    if plot:
        # Project in 2D for visualization
        rp = GaussianRandomProjection(n_components=2)
        projected = rp.fit_transform(X)
        plot_2d(projected, y)

        # Project in 3D for visualization
        rp = GaussianRandomProjection(n_components=3)
        projected = rp.fit_transform(X)
        plot_3d(projected, y)

    # # Plot eps vs. n components
    # eps_range = [0.4, 0.6, 0.8, 0.99]  # For Fashion MNIST eps 0.4 to 0.999 (must be < 1)
    # num_components = []
    # for eps in eps_range:
    #     rp = GaussianRandomProjection(n_components='auto', eps=eps)
    #     projected = rp.fit_transform(X)
    #     num_components.append(projected.shape)
    # print(num_components)

    # Determine min components for varying eps
    min_dims = []
    eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    for e in eps_range:
        min_dims.append(johnson_lindenstrauss_min_dim(n_samples=X.shape[0], eps=e))
    print('\nmin dims', min_dims)
    print('\nX shape:', X.shape)

    # Measure variation across multiple runs
    means_list = []
    stdev_list = []
    kurtosis_list = []
    iterations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in iterations:
        rp3 = GaussianRandomProjection(n_components=10)  # 10 components to help visualize the variation
        projected3 = rp3.fit_transform(X)
        means_list.append(np.mean(projected3))
        stdev_list.append(np.std(projected3))
        kurtosis_list.append(np.mean(kurtosis(projected3)))
        projected_df = pd.DataFrame(projected3)
        projected_df.to_csv('projected.csv')
        print(plot_kurtosis(projected3))

    # http://kitchingroup.cheme.cmu.edu/blog/2013/09/13/Plotting-two-datasets-with-very-different-scales/
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(iterations, means_list, label='Mean', color='red')
    ax1.plot(iterations, stdev_list, label='Std Deviation', color='blue')
    ax1.set_xlabel('Iteration', fontsize=18)
    ax1.legend()
    ax2 = ax1.twinx()
    ax2.plot(iterations, kurtosis_list, label='Kurtosis', color='green')
    plt.legend()
    plt.show()

    # print('\ncomponents_ shape:', rp3.components_.shape)

    # Project on to an 'optimal' number of components
    rp2 = GaussianRandomProjection(n_components=331)
    projected2_train = rp2.fit_transform(X)
    projected2_test = rp2.transform(X_test)
    print('\nRP projected X_train:', projected2_train.shape)

    return projected2_train, projected2_test


def tree_selection(X, y, dataset, X_test):

    # https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection

    # Determine feature importances
    clf = ExtraTreesClassifier(n_estimators=10, criterion='entropy')
    clf = clf.fit(X, y)
    # print(clf.feature_importances_)

    # Plot feature importances (heatmap)
    # https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances_faces.html
    importances = clf.feature_importances_
    # importances_shaped = importances.reshape(28, 28)
    # sns.heatmap(importances_shaped, cmap='hot')
    # plt.show()

    # Visualize clusters with only 3 features selected
    model = SelectFromModel(clf, prefit=True, threshold=-np.inf, max_features=3)
    projected = model.transform(X)
    plot_3d(projected, y)

    # Project onto optimal number of features (based on mean)
    model = SelectFromModel(clf, prefit=True, threshold='mean')  # 'mean' or 'median' for threshold
    projected_mean_train = model.transform(X)
    projected_mean_test = model.transform(X_test)
    print('\nTree-based selection projected X_train (mean):', projected_mean_train.shape)

    # Project onto optimal number of features (based on median)
    model = SelectFromModel(clf, prefit=True, threshold='median')  # 'mean' or 'median' for threshold
    projected_median = model.transform(X)
    print('\nTree-based selection projected X_train (median):', projected_median.shape)

    # Plot feature importances
    importances_sorted = -np.sort(-importances)
    # plt.plot(np.cumsum(importances_sorted))
    plt.plot(importances_sorted)
    plt.xlabel('Features', fontsize=18)
    plt.ylabel('Feature importance', fontsize=18)
    plt.title('Feature Importance', fontsize=22)
    plt.axvline(x=projected_mean_train.shape[1], color="red", linestyle="--", label='mean')
    plt.axvline(x=projected_median.shape[1], color="blue", linestyle="--", label='median')
    plt.legend()
    plt.show()

    pd.DataFrame(importances_sorted).to_csv("importances.csv")

    return projected_mean_train, projected_mean_test


def neural_network_classifier(X_train, y_train, X_test, y_test):

    class_labels = ['0 T-shirt/top', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt',
                    '7 Sneaker', '8 Bag', '9 Ankle boot']

    # class_labels = ['0 <=50K', '1 >50K']

    # Check train and test data set sizes
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # ANN
    # Optimal parameters from Assignment 1: Fashion MNIST
    classifier = MLPClassifier(hidden_layer_sizes=(1000,), activation='logistic', solver='sgd', alpha=100,
                               learning_rate='adaptive', learning_rate_init=0.001, max_iter=100, random_state=42)

    # Fit
    t0 = time.time()
    print('\nTime fit started:', time.strftime('%X %x %Z'))
    classifier.fit(X_train, y_train)
    print('Time fit ended:', time.strftime('%X %x %Z'))
    print('Fit done in %0.3fs' % (time.time() - t0))

    # Predict
    t0 = time.time()
    print('\nTime predict started:', time.strftime('%X %x %Z'))
    y_pred = classifier.predict(X_test)
    print('Time predict ended:', time.strftime('%X %x %Z'))
    print('Predict done in %0.3fs' % (time.time() - t0))

    # Print results
    print("\nClassification report:\n", classification_report(y_test, y_pred, target_names=class_labels))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred, labels=range(10)))  # TODO: 2 Adult, 10 Fashion


if __name__ == "__main__":

    # To run this, make sure the data files are in a sub-directory "./data".  Refer to README.txt for instructions.

    datasets = ['Fashion MNIST', 'Adult', 'blobs']
    dataset = datasets[0]
    training_size = 10000  # Use to further sub-sample the Train set if needed (0 for all data)
    verbose = False
    plot = True

    if dataset == 'Fashion MNIST':   # Load and pre-process Fashion MNIST data
        X_train, X_test, y_train, y_test = load_fashion_data(training_size)
        class_labels = ['0 T-shirt/top', '1 Trouser', '2 Pullover', '3 Dress', '4 Coat', '5 Sandal', '6 Shirt',
                        '7 Sneaker', '8 Bag', '9 Ankle boot']

    if dataset == 'Adult':   # Load and pre-process Adult data
        X_train, X_test, y_train, y_test = load_adult_data(training_size)
        class_labels = ['0 <=50K', '1 >50K']

    if dataset == 'blobs':
        X_train, y_train = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
        class_labels = ['0', '1', '2', '3']

    # 1. KMeans Clustering
    # projected_X_train, projected_X_test = kmeans_clustering(X_train, y_train, class_labels, dataset, verbose, X_test)

    # 2. Expectation Maximization Clustering (GMM)
    projected_X_train, projected_X_test = em_clustering(X_train, y_train, class_labels, dataset, verbose, X_test)

    # 3. PCA
    # projected_X_train, projected_X_test = pca_analysis(X_train, y_train, dataset, plot, X_test)

    # 4. ICA
    # projected_X_train, projected_X_test = ica_analysis(X_train, y_train, dataset, plot, X_test)

    # 5. Random Projection
    # projected_X_train, projected_X_test = rp_analysis(X_train, y_train, dataset, plot, X_test)

    # 6. Tree-based feature selection
    # projected_X_train, projected_X_test = tree_selection(X_train, y_train, dataset, X_test)

    # print('\nProjected X Train shape:', projected_X_train.shape)
    # print()

    # 7. Rerun KMeans Clustering after dimensionality reduction
    # kmeans_clustering(projected_X_train, y_train, class_labels, dataset, verbose, X_test)

    # 8. Rerun Expectation Maximization Clustering (GMM) after dimensionality reduction
    # em_clustering(projected_X_train, y_train, class_labels, dataset, verbose, X_test)

    # 9. Rerun assignment 1 neural network on newly projected data
    neural_network_classifier(projected_X_train, y_train, projected_X_test, y_test)


