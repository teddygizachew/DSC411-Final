import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# clean data for classification task
def clean_data(unclean_data):
    # drop fnlwgt column as it is irrelevant to the tasks
    # definition: number of people census believes entry represents
    # in other words, the amount of people that can be categorized by the set
    cleaned_data = unclean_data.drop(columns=['fnlwgt'])

    # replace missing values:
    # replace missing workclass and occupation with 'unemployed', as per analysis
    # missing values in dataframe is currently '?'
    cleaned_data['workclass'] = cleaned_data['workclass'].replace('?', 'Unemployed')
    cleaned_data['occupation'] = cleaned_data['occupation'].replace('?', 'Unemployed')

    # replace missing 'native_country' with the most common response
    # analysis returned no exact reasoning for missing values
    cleaned_data['native_country'] = cleaned_data['native_country'].replace('?', cleaned_data['native_country'].mode().iloc[0])

    # binning 'ages' into with pd.cut
    # tested with # of bins at 4, 5, 6, 7, 8, and 9
    # 7 bins separated ages into more common life-stages
    cleaned_data['age'] = pd.cut(cleaned_data['age'], 7)

    return cleaned_data


def main():
    # import data
    data = pd.read_csv("adult.csv", header=0)

    # clean data
    data_cleaned = clean_data(data)

    # export transformed data set to csv
    data_cleaned.to_csv("project.csv", sep=',')

    # export analysis document (to be implemented later)
    # analysis()


if __name__ == '__main__':
    main()
