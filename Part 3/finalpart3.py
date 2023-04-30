import time
from enum import Enum

import matplotlib.pyplot as plt
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier, plot_tree
import wittgenstein as wt
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns


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

        # Perform 5-fold cross-validation on the training set
        cv_scores = cross_val_score(dt, X_train, y_train, cv=5)
        print("Accuracy on 5-fold cross-validation:", cv_scores.mean())

        # Perform 10-fold cross-validation on the training set
        cv_scores = cross_val_score(dt, X_train, y_train, cv=10)
        print("Accuracy on 10-fold cross-validation:", cv_scores.mean())

        # Generate the classification report
        y_pred = dt.predict(X_test)
        report = classification_report(y_test, y_pred)
        print("Classification report:\n", report)

        # Visualize the decision tree
        plt.figure(figsize=(50, 10))
        plot_tree(dt, filled=True, feature_names=data.columns[:-1], class_names=["<=50K", ">50K"], fontsize=5)
        plt.show()


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

    # mpl.use("macosx")  # TkAgg

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


def convert_to_numerical_dbscan(data):
    # Convert categorical variables to numerical variables
    le = LabelEncoder()
    columns_to_encode = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country',
                         'census_income']

    for column in columns_to_encode:
        data[column] = le.fit_transform(data[column].astype(str))

    # Convert to numeric type data
    data[columns_to_encode] = data[columns_to_encode].apply(pd.to_numeric)

    # One-hot encoded data
    data_one_hot = data.copy()
    attributes = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country',
                  'census_income']
    data = pd.get_dummies(data_one_hot, columns=attributes, prefix=attributes)

    return data


def dbscan_outliers(data):
    # create a temporary hold dataframe for processing
    temp_data = convert_to_numerical_dbscan(data)

    """ Used to calculate epsilon for dbscan outlier detection
        Commented out because it takes a long time to iterate
        Already used 
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(hold_data)
    distances, indices = nbrs.kneighbors(hold_data)
    # Plotting K-distance Graph
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(figsize=(20, 10))
    plt.plot(distances)
    plt.title('K-distance Graph', fontsize=20)
    plt.xlabel('Data Points sorted by distance', fontsize=14)
    plt.ylabel('Epsilon', fontsize=14)
    plt.show()
    epsilon = 2000 """

    # use DBSCAN to generate labels for outlier y/n
    model = DBSCAN(eps=2000, min_samples=25).fit(temp_data)
    outliers_df = model.labels_

    data['dbscan_outliers'] = outliers_df

    outlier_plot_dbscan(data, "DBSCAN")

    return outliers_df


def outlier_plot_dbscan(data, outlier_method_name):
    print()
    print(f'Outlier Method: {outlier_method_name}')

    print(f"Number of non anomalous values  {len(data[data['dbscan_outliers'] != -1])}")

    num_outliers = (data['dbscan_outliers'] == -1).sum()
    num_total = data.shape[0]
    percent_outliers = (num_outliers / num_total) * 100

    print(f"Number of outliers detected: {num_outliers}")
    print(f'Total Number of Values: {len(data)}')
    print(f"Percentage of outliers: {percent_outliers:.2f}%")
    print()


def convert_to_numerical_iso_forest(data):
    # Perform one-hot encoding on the categorical variables (for classification purposes)
    cat_vars = ['age', 'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for var in cat_vars:
        cat_list = pd.get_dummies(data[var], prefix=var)
        data = data.join(cat_list)

    data = data.drop(cat_vars, axis=1)
    return data


def isolation_forest(data):
    data = convert_to_numerical_iso_forest(data)
    anomaly_inputs = ['hours_per_week', 'education_numeric']
    model = IsolationForest(contamination=float(0.1), random_state=42)
    model.fit(data[anomaly_inputs])

    data['iso_forest_anomaly_score'] = model.decision_function(data[anomaly_inputs])
    data['iso_forest_outliers'] = model.predict(data[anomaly_inputs])

    # export transformed data set to csv
    output_filename = "IsolationForest.csv"
    data.to_csv(output_filename, sep=',', index=False)

    outlier_plot(data, "IsolationForest")

    return model.predict(data[anomaly_inputs])


def outlier_plot(data, outlier_method_name):
    print(f'Outlier Method: {outlier_method_name}')

    print(f"Number of non anomalous values  {len(data[data['iso_forest_outliers'] == 1])}")

    num_outliers = (data['iso_forest_outliers'] == -1).sum()
    num_total = data.shape[0]
    percent_outliers = (num_outliers / num_total) * 100

    print(f"Number of outliers detected: {num_outliers}")
    print(f'Total Number of Values: {len(data)}')
    print(f"Percentage of outliers: {percent_outliers:.2f}%")
    print()


def rule_generation(data):
    print("Generating rules for 'income is >50K' via wittgenstein RIPPER...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, data['census_income'],
                                                        test_size=0.4, random_state=42)

    # train the model
    mdl = wt.RIPPER()
    mdl.fit(X_train, class_feat='census_income', pos_class=' <=50K')

    # isolate the target predictor and rest of the trained data for precision checking
    check_feature = X_test['census_income']
    X_test2 = X_test.drop(['census_income'], axis='columns')

    # print the generated rules
    print(mdl.ruleset_)
    mdl.ruleset_.out_pretty()

    # perform the precision and recall comparisons
    precision = mdl.score(X_test2, check_feature, precision_score)
    recall = mdl.score(X_test2, check_feature, recall_score)
    print("RIPPER Rule Precision for 'income is <=50K': " + str(round((precision * 100), 2)) + "%")
    print("RIPPER Rule Recall for 'income is <=50K': " + str(round((recall * 100), 2)) + "%")

    '''# train a second model to compare precision between the two predictors
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, data['census_income'],
                                                        test_size=0.4, random_state=40)
    mdl2 = wt.RIPPER()
    mdl2.fit(X_train, class_feat='census_income', pos_class=' >50K')

    # isolate the target predictor and rest of the trained data for precision checking
    check_feature = X_test['census_income']
    X_test2 = X_test.drop(['census_income'], axis='columns')

    # perform the prediction
    precision = mdl.score(X_test2, check_feature, precision_score)
    recall = mdl.score(X_test2, check_feature, recall_score)
    print()
    print("RIPPER Rule precision for 'income is >50K': " + str(round((precision * 100), 2)) + "%")
    print("RIPPER Rule recall for 'income is >50K': " + str(round((recall * 100), 2)) + "%")'''
    print()


def naive_bayes(data):
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)

    # Perform one-hot encoding on the categorical variables (for classification purposes)
    cat_vars = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for var in cat_vars:
        cat_list = pd.get_dummies(data[var], prefix=var)
        data = data.join(cat_list)

    data = data.drop(cat_vars, axis=1)

    # Drop class label
    data['census_income'] = data['census_income'].apply(lambda x: 0 if x == ' <=50K' else 1)

    x_train, x_test, y_train, y_test = train_test_split(data, data["census_income"], test_size=0.2)

    # gaussian classifier
    mdl = GaussianNB().fit(x_train, y_train)

    print(f"Model accuracy: {mdl.score(x_test, y_test) * 100:.2f}%")

    results = pd.DataFrame({'actual': y_test, 'predicted': mdl.predict(x_test)})
    print(results.head())

    # Predict on test data
    y_pred = mdl.predict(x_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion matrix for Naive Bayes classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print("\nTest Data:")
    print(y_test.head(7))
    print("\nPredictions:")
    print(mdl.predict(x_test)[:7])
    print("\nProbability of predictions:")
    print(mdl.predict_proba(x_test)[:7])
    print(f"\nCross val score: {cross_val_score(GaussianNB(), x_train, y_train, cv=3)}")


def main():
    data = read_data("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")

    # clean data
    data_cleaned = clean_data(data)
    # recreate for holding
    data = data_cleaned.copy()

    # classify data
    classify(data_cleaned)

    # cluster
    cluster(data_cleaned, method=ClusteringMethod.ELBOW)

    # run DBSCAN outlier detection
    dbscan_outliers_list = dbscan_outliers(data_cleaned)

    iso_forest_list = isolation_forest(data_cleaned)

    # add outlier labels to output dataset for side-by-side comparison
    data['dbscan_outliers'] = dbscan_outliers_list
    data['iso_forest_outliers'] = iso_forest_list

    # run the rule-based classification task
    rule_generation(data)

    # run the naive bayes classification
    naive_bayes(data)

    # export transformed data set to csv
    output_filename = "projectPart3Output.csv"
    data.to_csv(output_filename, sep=',', index=False)
    print(f"Exported cleaned data to {output_filename}")


if __name__ == '__main__':
    main()
