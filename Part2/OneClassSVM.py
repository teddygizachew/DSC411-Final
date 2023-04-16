import pandas as pd
from sklearn.ensemble import IsolationForest


def read_data(file_path):
    data = pd.read_csv(file_path, header=0)
    data = data.drop(['census_income'], axis=1, inplace=False)
    return data


def convert_to_numerical(data):
    # Perform one-hot encoding on the categorical variables (for classification purposes)
    cat_vars = ['age', 'workclass', 'mariatal_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    for var in cat_vars:
        cat_list = pd.get_dummies(data[var], prefix=var)
        data = data.join(cat_list)

    data = data.drop(cat_vars, axis=1)
    return data


def isolation_forest(data):
    anomaly_inputs = ['hours_per_week', 'education_numeric']
    model = IsolationForest(contamination=float(0.1), random_state=42)
    model.fit(data[anomaly_inputs])

    data['anomaly_scores'] = model.decision_function(data[anomaly_inputs])
    data['anomaly'] = model.predict(data[anomaly_inputs])

    # export transformed data set to csv
    output_filename = "IsolationForest.csv"
    data.to_csv(output_filename, sep=',', index=False)

    outlier_plot(data, "IsolationForest")


def outlier_plot(data, outlier_method_name):
    print(f'Outlier Method: {outlier_method_name}')

    print(f"Number of non anomalous values  {len(data[data['anomaly'] == 1])}")

    num_outliers = (data['anomaly'] == -1).sum()
    num_total = data.shape[0]
    percent_outliers = (num_outliers / num_total) * 100

    print(f"Number of outliers detected: {num_outliers}")
    print(f'Total Number of Values: {len(data)}')
    print(f"Percentage of outliers detected: {percent_outliers:.2f}%")


def main():
    file_path = "../Part1/projectPart1Output.csv"
    data = read_data(file_path)

    data = convert_to_numerical(data)

    isolation_forest(data)


if __name__ == '__main__':
    main()
