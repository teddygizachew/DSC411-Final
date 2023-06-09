from enum import Enum
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from kneed import KneeLocator
from sklearn.preprocessing import normalize
import matplotlib as mpl


class ClusteringMethod(Enum):
    ELBOW = "Elbow"
    SILHOUETTE = "Silhouette"


def read_data(filepath):
    # import data
    data = pd.read_csv(filepath, header=None,
                       names=['age', 'workclass', 'fnlwgt', 'education', 'education_numeric', 'marital_status',
                              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                              'hours_per_week', 'native_country', 'census_income'])
    return data


# clean data for classification task
def clean_data(unclean_data):
    # drop 'education', as it is a redundant feature with 'education-numeric' when normalized
    cleaned_data = unclean_data.drop(columns=['education'])

    # replace missing values:
    # replace missing workclass and occupation with 'unemployed', as per analysis
    # missing values in dataframe is currently '?'
    cleaned_data['workclass'] = cleaned_data['workclass'].replace('?', 'Unemployed')
    cleaned_data['occupation'] = cleaned_data['occupation'].replace('?', 'Unemployed')

    # replace missing 'native_country' with a new value called 'other'
    # analysis returned no exact reasoning for missing values
    cleaned_data['native_country'] = cleaned_data['native_country'].replace(' ?', 'Other')

    return cleaned_data


def classify(data):
    # Perform one-hot encoding on the categorical variables (for classification purposes)
    cat_vars = ['age', 'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for var in cat_vars:
        cat_list = pd.get_dummies(data[var], prefix=var)
        data = data.join(cat_list)

    data = data.drop(cat_vars, axis=1)

    # Set the test sizes to use
    test_sizes = [0.2, 0.4, 0.5]

    for test_size in test_sizes:
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data.drop('census_income', axis=1), data['census_income'],
                                                            test_size=test_size, random_state=42)

        # Train a decision tree on the training set
        dt = DecisionTreeClassifier(random_state=42, max_depth=7)
        dt.fit(X_train, y_train)

        # Evaluate the model on the testing set
        print(f"Split: {100 - (100 * test_size)} - {100 * test_size}")
        print("Accuracy on test set:", dt.score(X_test, y_test))

        # Generate the classification report
        y_pred = dt.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

        # Visualize the decision tree
        plt.figure(figsize=(50, 10))
        plot_tree(dt, filled=True, feature_names=data.columns[:-1], class_names=["<=50K", ">50K"], fontsize=5)
        plt.show()
        
    # Perform 5-fold cross-validation on the entire dataset
    cv_scores_5 = cross_val_score(dt, data.drop('census_income', axis=1), data['census_income'], cv=5)
    print("Accuracy on 5-fold cross-validation:", cv_scores_5.mean())

    # Perform 10-fold cross-validation on the entire dataset
    cv_scores_10 = cross_val_score(dt, data.drop('census_income', axis=1), data['census_income'], cv=10)
    print("Accuracy on 10-fold cross-validation:", cv_scores_10.mean())


def cluster(data, method):
    """
    Elbow Method Execution Time: --- 14.488507986068726 seconds ---
    Silhouette Execution Time: --- 192.42863178253174 seconds ---
    """
    # Drop class label
    data = data.drop(["census_income"], axis=1)

    features_normalized = convert_to_numerical(data)
    if method == ClusteringMethod.ELBOW:
        start_time = time.time()
        k = elbow_method(features_normalized)
        print("Elbow Method Execution Time: --- %s seconds ---" % (time.time() - start_time))
        print(f'Elbow Method k -> {k}')
    elif method == ClusteringMethod.SILHOUETTE:
        start_time = time.time()
        k = silhouette_method(features_normalized)
        print("Silhouette Execution Time: --- %s seconds ---" % (time.time() - start_time))
        print(f'Silhouette k -> {k}')
    else:
        print(f'Cluster method {method} does not exist!')

    visualize(k=2, features_normalized=features_normalized)
    visualize(k=3, features_normalized=features_normalized)


# Create a 3D figure
def visualize(k, features_normalized):
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300)
    kmeans.fit(features_normalized)

    mpl.use("macosx")  # TkAgg

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the clusters in 3D
    ax.scatter3D(features_normalized[:, 0], features_normalized[:, 1], features_normalized[:, 2], c=kmeans.labels_)
    ax.scatter3D(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=100,
                 c='red', label='Centroids')
    ax.set_xlabel('Age')
    ax.set_ylabel('Education')
    ax.set_zlabel('Hours Per Week')
    plt.legend()
    plt.show()

    # get the SSE
    print(kmeans.inertia_)

    # get the centroids
    print(kmeans.cluster_centers_)


def convert_to_numerical(data):
    # Convert categorical variables to numerical variables
    le = LabelEncoder()
    columns_to_encode = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']

    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column].astype(str))

    # Convert to numeric type data
    data[columns_to_encode] = data[columns_to_encode].apply(pd.to_numeric)

    # One-hot encoded data
    data_one_hot = data.copy()
    attributes = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    data = pd.get_dummies(data_one_hot, columns=attributes, prefix=attributes)

    # Standardize numerical variables
    normalized_data = normalize(data)

    return normalized_data


def silhouette_method(data):
    # Store the silhouette scores for each k in a dictionary
    silhouette_scores = {}

    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        # compute the silhouette score for the clustering
        score = silhouette_score(data, kmeans.labels_)

        # store the silhouette score for the current k value
        silhouette_scores[k] = score

    # plot the silhouette scores for each k
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()))
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()

    # Optimal number of clusters based on the silhouette score
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    return best_k


def elbow_method(data):
    sse = []

    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, init="k-means++")
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    # plot the graph

    plt.plot(range(1, 10), sse)
    plt.xlabel("Number of clusters")
    plt.ylabel("SSE")
    plt.show()

    # auto reading the knee of the plot
    kl = KneeLocator(range(1, 10), sse, curve="convex", direction="decreasing")

    print(sse)
    print(kl.elbow)
    return kl.elbow


def main():
    data = read_data("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")

    # clean data
    data_cleaned = clean_data(data)

    # classify data
    classify(data_cleaned)

    # cluster
    cluster(data_cleaned, method=ClusteringMethod.ELBOW)

    # export transformed data set to csv
    output_filename = "projectPart1Output.csv"
    data_cleaned.to_csv(output_filename, sep=',', index=False)
    print(f"Exported cleaned data to {output_filename}")


if __name__ == '__main__':
    main()
