import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


head = st.container()
data = st.container()
model_sec = st.container()
footer = st.container()


@st.cache
def read_data(file):
    return pd.read_csv(file)


@st.cache()
def train_model(model):
    if model == 'Decision Tree Regressor':
        reg = DecisionTreeRegressor(max_depth=max_depth)
        reg.fit(x_train, y_train)
        score = reg.score(x_test, y_test)
        mae = mean_absolute_error(y_test, reg.predict(x_test))
        fi = reg.feature_importances_
        return score, mae, fi
    else:
        reg = RandomForestRegressor(
            max_depth=max_depth, n_estimators=n_estimators)
        reg.fit(x_train, y_train)
        score = reg.score(x_test, y_test)
        mae = mean_absolute_error(y_test, reg.predict(x_test))
        fi = reg.feature_importances_
        return score, mae, fi


with head:
    st.title('Vehicle Price Prediction')
    st.write(
        'The **goal** of this project is to make a model which can predict the price of a vehicle.')
    st.image('car.jpg')


with data:
    st.header('Vehicle Dataset')
    st.write('The data includes Vehicle informations, such as the model, manufacturer,condition, size, color, etc.')
    st.write("Let's see the first 5 rows of  the dataset:")
    df = read_data('data/veh_sample2.csv')
    st.write(df.head())

    st.header('EDA')
    st.write('1)    The most expensive manufacturers')

    df = df[df['price'] <= df['price'].quantile(0.99)]

    grouped_by_man = df.groupby('manufacturer')[
        'price'].median()[:5].sort_values(ascending=False)
    st.table(grouped_by_man)

    st.write('2) Number of vehicles per fuel type')

    grouped_by_fuel = df['fuel'].value_counts()

    st.bar_chart(grouped_by_fuel)

    st.write('3) Price distribution of the vehicles')

    fig, ax = plt.subplots()
    ax.hist(df['price'], bins=20, edgecolor="k")
    plt.xlabel('Price')
    plt.ylabel('Frequency')

    st.pyplot(fig)

    st.write('4) Number of vehicles per condition')

    grouped_by_condition = df['condition'].value_counts()

    st.bar_chart(grouped_by_condition)

    # data preprocessing
    df.drop(['county', 'url', 'region_url', 'image_url', 'vin', 'size', 'condition', 'id', 'region',
            'model', 'type', 'paint_color', 'lat', 'long', 'description'], axis=1, inplace=True)
    df_preprocessing = df.copy()
    df_preprocessing = df_preprocessing[df_preprocessing['price'] >= 799]
    df_preprocessing = df_preprocessing[df_preprocessing['odometer']
                                        <= df_preprocessing['odometer'].quantile(0.99)]

    df_preprocessing = df_preprocessing[df_preprocessing['cylinders'].isin(
        ['4 cylinders', '6 cylinders', '8 cylinders', np.nan])]
    df_preprocessing.cylinders.fillna('5.5', inplace=True)
    df_preprocessing['cylinders'] = df_preprocessing['cylinders'].map(
        {'5.5': 5.5, '4 cylinders': 4, '6 cylinders': 6, '8 cylinders': 8})

    dummies_for_state = pd.get_dummies(
        df_preprocessing.state, drop_first=True, prefix='state')
    df_preprocessing = pd.concat([df_preprocessing, dummies_for_state], axis=1)
    df_preprocessing.drop('state', axis=1, inplace=True)

    df_preprocessing = df_preprocessing[df_preprocessing['manufacturer'].isin(['ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep', 'gmc',
                                                                               'dodge', 'ram', 'hyundai', 'subaru', 'bmw', 'kia',
                                                                               'mercedes-benz', 'volkswagen', 'chrysler', 'buick', 'cadillac',
                                                                               'lexus', 'mazda', 'audi'])]
    dummies_for_manufacturer = pd.get_dummies(
        df_preprocessing.manufacturer, drop_first=True)
    df_preprocessing = pd.concat(
        [df_preprocessing, dummies_for_manufacturer], axis=1)
    df_preprocessing.drop('manufacturer', axis=1, inplace=True)

    df_preprocessing = df_preprocessing[df_preprocessing['fuel'].isin([
                                                                      'gas', 'diesel'])]
    df_preprocessing['fuel'] = df_preprocessing['fuel'].map(
        {'gas': 0, 'diesel': 1})

    df_preprocessing = df_preprocessing[df_preprocessing['title_status'].isin([
                                                                              'clean', 'rebuilt'])]
    df_preprocessing['title_status'] = df_preprocessing['title_status'].map(
        {'clean': 0, 'rebuilt': 1})

    df_preprocessing = df_preprocessing[df_preprocessing['transmission'].isin([
                                                                              'automatic', 'manual'])]
    df_preprocessing['transmission'] = df_preprocessing['transmission'].map(
        {'automatic': 0, 'manual': 1})

    df_preprocessing = df_preprocessing[df_preprocessing['drive'].notna()]

    dummies_for_drive = pd.get_dummies(df_preprocessing.drive, drop_first=True)
    df_preprocessing = pd.concat([df_preprocessing, dummies_for_drive], axis=1)
    df_preprocessing.drop('drive', axis=1, inplace=True)

    x = df_preprocessing.drop('price', axis=1)
    y = df_preprocessing['price']

    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=101)


with model_sec:
    st.header('Model Training')
    st.text('There are 2 available models:')
    st.markdown('* Decision Tree Regressor')
    st.markdown('* Random Forest Regressor')
    st.write('Here you can select a model,choose its hyperparameters and see how the performance changes.')
    st.markdown(
        '**Important Note**: The training process may take a while.')

    sel_col, displ_col = st.columns(2)

    model = sel_col.selectbox('Select a model', options=[
                              'Decision Tree Regressor', 'Random Forest Regressor'], index=0)

    max_depth = sel_col.slider('What should be the max_depth of the model?',
                               min_value=10, max_value=100, value=20, step=10)
    if model != 'Decision Tree Regressor':
        n_estimators = sel_col.selectbox('How many trees should be there?', options=[
                                         50, 100, 150], index=0)

    # model training
    score, mae, fi = train_model(model)
    displ_col.subheader('R-squared score')
    displ_col.text(score)
    displ_col.subheader('Mean Absolute Error')
    displ_col.text(mae)

    features_df = pd.DataFrame(data=fi, index=x.columns.values, columns=[
                               'feature importance']).sort_values(by='feature importance', ascending=False)[:5]
    model_sec.subheader('Feature Importances')
    model_sec.table(features_df)
    model_sec.markdown(
        'As we can see the most important features for predicting the price of a car are: `year`, the wheel drive (`fwd`/`rwd`), `odometer`, `cylinders` and the `fuel`.')
with footer:
    st.header('Conclusion')
    st.write(
        'Overall, Random Forest Regressor is the best model, with almost 83.7% accuracy.')
    st.text('*This result was achieved, by having the max_depth=40 and trees=100*')
