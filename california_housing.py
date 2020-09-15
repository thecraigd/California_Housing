import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


# Title - The title and introductory text and images are all written in Markdown format here, using st.write()

st.write("""
[![craigdoesdata logo][logo]][link]
[logo]: https://www.craigdoesdata.de/img/logo/logo_w_sm.gif
[link]: https://www.craigdoesdata.de/

# California Housing Prices

![Some California Houses](https://images.pexels.com/photos/2401665/pexels-photo-2401665.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260)

Photo by [Leon Macapagal](https://www.pexels.com/@imagevain?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/aerial-photography-of-concrete-houses-2401665/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels).

------------

This app predicts California Housing Prices using a machine learning model powered by [Scikit Learn](https://scikit-learn.org/).

The data for the model is the famous [California Housing Prices](https://www.kaggle.com/camnugent/california-housing-prices) Dataset.

Play with the values via the sliders on the left panel to generate new predictions.
""")
st.write("---")


# Import Data - the original source has been commented out here, but left in so the CSV files can be sourced again in future, if needed.


# Import the data from Google
# train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
# train_df = train_df.reindex(np.random.permutation(train_df.index)) # randomise the examples
# test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# full_df = pd.concat([train_df, test_df])

train_df = pd.read_csv('data/train_df.csv')
test_df = pd.read_csv('data/test_df.csv')
full_df = pd.read_csv('data/full_df.csv')

# Assign features (X) and target (Y) to DataFrames
X = full_df
Y = X.pop('median_house_value')

# Sidebar - this sidebar allows the user to set the parameters that will be used by the model to create the prediction.
st.sidebar.header('Specify Input Parameters - these will determine the predicted value.')

def features_from_user():
    longitude = st.sidebar.slider('Longitude', float(full_df.longitude.min()), float(full_df.longitude.max()), float(full_df.longitude.mean()))
    latitude = st.sidebar.slider('Latitude', float(full_df.latitude.min()), float(full_df.latitude.max()), float(full_df.latitude.mean()))
    housing_median_age = st.sidebar.slider('Housing Median Age', float(full_df.housing_median_age.min()), float(full_df.housing_median_age.max()), float(full_df.housing_median_age.mean()))
    total_rooms = st.sidebar.slider('Total Rooms', float(full_df.total_rooms.min()), float(full_df.total_rooms.max()), float(full_df.total_rooms.mean()))
    total_bedrooms = st.sidebar.slider('Total Bedrooms', float(full_df.total_bedrooms.min()), float(full_df.total_bedrooms.max()), float(full_df.total_bedrooms.mean()))
    population = st.sidebar.slider('Population', float(full_df.population.min()), float(full_df.population.max()), float(full_df.population.mean()))
    households = st.sidebar.slider('Households', float(full_df.households.min()), float(full_df.households.max()), float(full_df.households.mean()))
    median_income = st.sidebar.slider('Median Income', float(full_df.median_income.min()), float(full_df.median_income.max()), float(full_df.median_income.mean()))
    
    data = {'Longitude': longitude,
            'Latitude': latitude,
            'Housing Median Age': housing_median_age,
            'Total Rooms': total_rooms,
            'Total Bedrooms': total_bedrooms,
            'Population': population,
            'Households': households,
            'Median Income': median_income}

    features = pd.DataFrame(data, index = [0])
    return features

df = features_from_user()

# Display specified input parameters
st.write('Specified Input Parameters:')
st.table(df)
st.write('---')

X = X.drop('Unnamed: 0', axis=1)


# Build Regression Model - the 3 lines below are commented out to save processing time.
# These lines would allow us to re-run the model whenever required.

# model = RandomForestRegressor()
# model.fit(X, Y)
# dump(model, 'model.joblib') 

# Load the saved model
model_new = load('data/model.joblib') 

# Apply Model to Make Prediction
prediction = int(model_new.predict(df))
prediction_nice = f"{prediction:,d}"

# Main Panel - display prediction

st.header('Prediction of Median House Value:')
st.write('Based on your selections, the model predicts a value of %s US Dollars.' % prediction_nice)
st.write('---')
