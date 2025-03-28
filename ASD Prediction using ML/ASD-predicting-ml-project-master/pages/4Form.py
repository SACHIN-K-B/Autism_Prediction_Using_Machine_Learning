import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer

st.title(":bookmark_tabs: :blue[Autism data assessment]")
st.write("---")
st.write("Fill the form below to check if your child is suffering from ASD ")

# Load dataset
autism_dataset = pd.read_csv('data_csv.csv')

# Separating the data and labels
X = autism_dataset.drop(columns=['ASD_traits', 'Ethnicity', 'Who_completed_the_test', 'Qchat_10_Score', 'Childhood Autism Rating Scale', "CASE_NO_PATIENT'S"], axis=1)
Y = autism_dataset['ASD_traits']

# Define numerical and categorical columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Create a preprocessing and classification pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', SVC())])

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

# Helper functions for form input transformation
def ValueCount(str):
    return 1 if str == "Yes" else 0

def Sex(str):
    return 1 if str == "Female" else 0

# Form layout goes here...
d1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
val1 = st.selectbox("Social Responsiveness", d1)

d2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
val2 = st.selectbox("Age", d2)

d3 = ["No", "Yes"]

valA1 = ValueCount(st.selectbox("A1 : I often notice small sounds when others do not", d3))
valA2 = ValueCount(st.selectbox("A2 : I usually concentrate more on the whole picture, rather than the small details", d3))
valA3 = ValueCount(st.selectbox("A3 : I find it easy to do more than one thing at once", d3))
valA4 = ValueCount(st.selectbox("A4 : If there is an interruption, I can switch back to what I was doing very quickly", d3))
valA5 = ValueCount(st.selectbox("A5 : I find it easy to read between the lines when someone is talking to me", d3))
valA6 = ValueCount(st.selectbox("A6 : I know how to tell if someone listening to me is getting bored", d3))
valA7 = ValueCount(st.selectbox("A7 : When I’m reading a story, Ifind it difficult to work out the character’s intention", d3))
valA8 = ValueCount(st.selectbox("A8 : I like to collect information about categories of things (e.g. types of car, types of bird, types of train, types of plant etc).", d3))
valA9 = ValueCount(st.selectbox("A9 : I find it easy to work out what someone is thinkig or feeling just by looking at their face.", d3))
valA10 = ValueCount(st.selectbox("A10 : I find it difficult to work out people’s intentions.", d3))

val3 = ValueCount(st.selectbox("Speech delay", d3))
val4 = ValueCount(st.selectbox("Learning disorder", d3))
val5 = ValueCount(st.selectbox("Genetic disorders", d3))
val6 = ValueCount(st.selectbox("Depression", d3))
val7 = ValueCount(st.selectbox("Intellectual disability", d3))
val8 = ValueCount(st.selectbox("Social/Behavioural issues", d3))
val9 = ValueCount(st.selectbox("Anxiety disorder", d3))

d4 = ["Female", "Male"]
val10 = Sex(st.selectbox("Gender", d4))

val11 = ValueCount(st.selectbox("Suffers from Jaundice", d3))
val12 = ValueCount(st.selectbox("Family member history with ASD", d3))

input_data = [valA1, valA2, valA3, valA4, valA5, valA6, valA7, valA8, valA9, valA10, val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12]

column_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient', 'Social_Responsiveness_Scale', 'Age_Years', 'Speech Delay/Language Disorder', 'Learning disorder', 'Genetic_Disorders', 'Depression', 'Global developmental delay/intellectual disability', 'Social/Behavioural Issues', 'Anxiety_disorder', 'Sex', 'Jaundice', 'Family_mem_with_ASD']

input_data_as_df = pd.DataFrame([input_data], columns=column_names)

# Transform the input data using the pipeline
std_data = model['preprocessor'].transform(input_data_as_df)

# Make a prediction using the classifier
prediction = model['classifier'].predict(std_data)

with st.expander("Analyze provided data"):
    st.subheader("Results:")
    if prediction == ['Yes']:
        st.info('The person is with Autism spectrum disorder')
    else:
        st.warning('The person is not with Autism spectrum disorder')