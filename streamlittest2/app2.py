import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import hashlib
from login import login_page
from register import register_page

# Set the page configuration
st.set_page_config(page_title="AutoML Assistant", layout="centered")

# Apply custom CSS styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1b2a;
        color: white;
        font-family: Arial, sans-serif;
    }
    .top-header {
        font-size: 2em;
        color: #e0e1dd;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .stRadio>div>label {
        display: flex;
        justify-content: center;
        padding: 10px;
        color: white;
        cursor: pointer;
    }
    .stRadio>div>label[data-baseweb="radio"]:hover {
        background-color: #444;
    }
    .stButton>button {
        color: #fff;
        background-color: #1b263b;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #e09500;
    }
    .stTextInput>div>input, .stFileUpload>div>div>div>div>button>span {
        background-color: #2c2c2c;
        color: #fff;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #ffae00;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    .main {
        background-color: #0d1b2a;
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .top-header {
        font-size: 2.5em;
        color: #ffae00;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 30px;
    }
    .stRadio>div>label {
        display: flex;
        justify-content: center;
        padding: 15px;
        color: #ffae00;
        cursor: pointer;
        font-size: 1.2em;
        border-radius: 5px;
        background: #1b263b;
        border: 1px solid #444;
    }
    .stRadio>div>label[data-baseweb="radio"]:hover {
        background-color: #e09500;
    }
    .stButton>button {
        color: #fff;
        background-color: #1b263b;
        border: none;
        padding: 12px 25px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ffae00;
        color: #1b263b;
    }
    .stTextInput>div>input, .stFileUpload>div>div>div>div>button>span {
        background-color: #2c2c2c;
        color: #fff;
        border: 1px solid #444;
        border-radius: 5px;
        font-size: 16px;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #ffae00;
    }
    .stMarkdown p {
        color: #d0d0d0;
    }
    .tabs-container {
        display: flex;
        justify-content: space-around;
        margin-bottom: 20px;
    }
    .tabs-container > div {
        flex: 1;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'users' not in st.session_state:
    st.session_state['users'] = {}

def home():
    if st.session_state['logged_in']:
        st.write("You are logged in!")
        st.markdown('<div class="top-header">AUTO ML</div>', unsafe_allow_html=True)

        # Define the navigation buttons
        tabs = ["Data Ingestion", "Data Transformation", "Auto Train ML models", "Freeze the learnings"]
        selected_tab = st.radio("Navigation", tabs, index=0, horizontal=True, format_func=lambda x: f"{x}")

        # Initialize a session state to store the dataset
        if 'dataset' not in st.session_state:
            st.session_state['dataset'] = None

        # Display content based on the selected tab
        if selected_tab == "Data Ingestion":
            st.header("Data Ingestion")
            
            # Choose data input method
            input_method = st.radio("Select data input method:", ["Enter file path", "Upload file"], horizontal=True)
            
            if input_method == "Enter file path":
                file_path = st.text_input("Path of the file", placeholder="Enter file path")
                file_name = st.text_input("Name of the file", placeholder="Enter file name with extension")
                if file_path and file_name:
                    try:
                        st.session_state['dataset'] = pd.read_csv(file_path)
                        st.success("File ingested successfully!")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            elif input_method == "Upload file":
                uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
                if uploaded_file is not None:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            st.session_state['dataset'] = pd.read_csv(uploaded_file)
                        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                            st.session_state['dataset'] = pd.read_excel(uploaded_file)
                        st.success("File ingested successfully!")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")

            if st.session_state['dataset'] is not None:
                st.write("Data Preview:")
                st.write(st.session_state['dataset'].head())

            st.subheader("Data dimensions:")
            if st.session_state['dataset'] is not None:
                st.text_input("Number of rows", disabled=True, value=str(st.session_state['dataset'].shape[0]))
                st.text_input("Number of columns", disabled=True, value=str(st.session_state['dataset'].shape[1]))

        elif selected_tab == "Data Transformation":
            st.header("Data Transformation")



                # columns = df.columns.tolist()
            # options = ['Option 1', 'Option 2', 'Option 3', 'Option 4']
            if st.session_state['dataset'] is not None:
                df = st.session_state['dataset']
                columns = df.columns.tolist()
                
                selected_options = st.multiselect('Select one or more options:', columns,key='multiselect_remove')
                if st.button('Remove features'):
                    #print(selected_options)
                    df = df.drop(columns=selected_options)
                    #st.session_state['dataset'] = df
                    st.success("Features removed!")

                if st.button('Remove duplicate fields'):
                    df = df.drop_duplicates()
                # st.write("test:")
                # missing_value_options = ['Remove rows with missing value','Replace missing value with mean','Replace missing value with median',
                #                          'Replace missing value with mode']
                # missingvaluechoice = st.selectbox('Handling missing values: ',missing_value_options,key='handling_missing_value')
                # missing_values = df.isnull().sum()
                    

                # if st.button('Handle missing values'):
                #     if missingvaluechoice == missing_value_options[0]:
                #         df = df.dropna()
                #     elif missingvaluechoice == missing_value_options[1]:
                #         df['marks'].fillna(df['marks'].mean(), inplace=True)

                columns = df.columns.tolist()
                columnchoice = st.selectbox('Select the column for which you want to remove the row with missing value',columns,key='remove_rows_missing_value')
                if st.button('Run',key='a'):
                    df = df.dropna(subset=[columnchoice])
                columnchoice = st.selectbox('Select the column for which you want to replace the missing value with mean',columns,key='replace_with_mean')
                if st.button('Run',key='b'):
                    df[columnchoice].fillna(df[columnchoice].mean(), inplace=True)
                columnchoice = st.selectbox('Select the column for which you want to replace the missing value with median',columns,key='replace_with_median')
                if st.button('Run',key='c'):
                    df[columnchoice].fillna(df[columnchoice].median(), inplace=True)
                columnchoice = st.selectbox('Select the column for which you want to replace the missing value with mode',columns,key='replace_with_mode')
                if st.button('Run',key='d'):
                    df[columnchoice].fillna(df[columnchoice].mode(), inplace=True)


                columns = df.columns.tolist()
                columns_to_convert = st.multiselect('Select one or more options:', columns,key='multiselect_other')
                if st.button('Convert features'):
                    if columns_to_convert:
                        # encoded_data = pd.get_dummies(df, columns_to_convert)
                        for col in columns_to_convert:

                            df[col] = pd.Categorical(df[col]).codes
                        #df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
                        st.success(f"Columns {columns_to_convert} converted to numeric!")

                columns = df.columns.tolist()
                targetvariable = st.selectbox('Select one or more options:', columns,key='targetvariable')
                if (st.button('target variable')):
                    st.success(f" {targetvariable} target variable set!")
                st.write(df.head())
            # selected_options = st.multiselect('Select one or more options:', columnlist)

                st.subheader("Data dimensions:")
                if st.button('Run'):
                    st.session_state['dataset'] = df
                    st.text_input("Number of rows", disabled=True, value=str(st.session_state['dataset'].shape[0]))
                    st.text_input("Number of columns", disabled=True, value=str(st.session_state['dataset'].shape[1]))

                st.session_state['dataset'] = df

        elif selected_tab == "Auto Train ML models":
            tabs = ["SVM", "Random Forest", "Decision Tree", "Stacking"]
            selected_tab = st.radio("Choose model", tabs, index=0, horizontal=True, format_func=lambda x: f"{x}",key='aa')

        # Title of the app
            if selected_tab==tabs[0]:
                st.title('Train Test Split Data')

                # Sliders for Training Data Split and Testing Data Split
                training_split = st.slider('Training Data Split', 0, 100, 70)
                testing_split = 100 - training_split

                # Display the splits
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Training Data Split:')
                    st.write(training_split)
                with col2:
                    st.write('Testing Data Split:')
                    st.write(testing_split)

                # Split Data button
                if st.button('Split Data'):
                    st.write(f'Training Data: {training_split}%')
                    st.write(f'Testing Data: {testing_split}%')
                # Title and description
                st.title("SVM Regression")
                st.write("""
                This app allows you to upload a dataset (CSV or Excel), specify the target variable, and train an SVM regression model.
                You can also evaluate the model's performance using R^2 and RMSE.
                """)

                

                if st.session_state['dataset'] is not None:
                    df = st.session_state['dataset']
                    if df is not None:
                        st.write("Dataset Preview:")
                        st.write(df.head())

                        # Select target variable
                        target_column = st.selectbox("Select the target column", df.columns)

                        if target_column:
                            X = df.drop(columns=[target_column])
                            y = df[target_column]

                            # Check and convert non-numeric columns in features
                            if not X.select_dtypes(include=[float, int]).shape[1] == X.shape[1]:
                                st.write("Non-numeric columns detected in features. Converting...")
                                X = pd.get_dummies(X)

                            # Check and convert non-numeric target
                            if y.dtype == 'object':
                                st.write("Non-numeric target column detected. Converting...")
                                le = LabelEncoder()
                                y = le.fit_transform(y)

                            # Print shapes and types to diagnose issues
                            st.write("Features (X) shape:", X.shape)
                            st.write("Target (y) shape:", y.shape)
                            st.write("Features (X) types:\n", X.dtypes)
                            st.write("Target (y) type:", y.dtype)

                            # Ensure there are no missing values
                            if X.isnull().values.any() or pd.Series(y).isnull().values.any():
                                st.write("Error: Missing values detected. Please clean your data.")
                            else:
                                # Convert X to numpy array
                                X = X.values

                                # Train-test split
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_split/100, random_state=42)

                                # SVR parameters
                                kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                                C = st.slider("C (Regularization parameter)", min_value=0.01, max_value=100.0, value=1.0, step=0.01)
                                epsilon = st.slider("Epsilon (Tube size)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
                                gamma = st.selectbox("Gamma", ["scale", "auto"])

                                # Train SVM model
                                model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
                                model.fit(X_train, y_train)

                                # Make predictions
                                y_pred = model.predict(X_test)

                                # Calculate performance metrics
                                r2 = r2_score(y_test, y_pred)
                                rmse = mean_squared_error(y_test, y_pred, squared=False)

                                # Display performance metrics
                                st.write("### Model Performance")
                                st.write(f"R^2 Score: {r2:.2f}")
                                st.write(f"RMSE: {rmse:.2f}")

                                # Display scatter plot of actual vs predicted values
                                st.write("### Actual vs Predicted Values")
                                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                                st.line_chart(results_df)


                                model_path = 'Models/SVM_model.pkl'
                                # Button to save the model
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                if st.button('Save SVM regression Model'):
                                    with open(model_path, 'wb') as file:
                                        pickle.dump(model, file)

                                    st.write(f'Model will be saved to: {model_path}')
                                    st.success('Model saved successfully!')


            elif selected_tab==tabs[1]:
        # Title and description
                st.title('Train Test Split Data')

                # Sliders for Training Data Split and Testing Data Split
                training_split = st.slider('Training Data Split', 0, 100, 70)
                testing_split = 100 - training_split

                # Display the splits
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Training Data Split:')
                    st.write(training_split)
                with col2:
                    st.write('Testing Data Split:')
                    st.write(testing_split)

                # Split Data button
                if st.button('Split Data'):
                    st.write(f'Training Data: {training_split}%')
                    st.write(f'Testing Data: {testing_split}%')

                st.title("Random Forest Model")
                st.write("""
                This app allows you to upload a dataset (CSV or Excel), specify the target variable, and train a Random Forest model.
                You can also evaluate the model's performance using Accuracy (for classification) and R^2 / RMSE (for regression).
                """)

                if st.session_state['dataset'] is not None:
                    df = st.session_state['dataset']

                    if df is not None:
                        st.write("Dataset Preview:")
                        st.write(df.head())

                        # Select target variable
                        target_column = st.selectbox("Select the target column", df.columns)

                        if target_column:
                            X = df.drop(columns=[target_column])
                            y = df[target_column]

                            # Check and convert non-numeric columns in features
                            if not X.select_dtypes(include=[float, int]).shape[1] == X.shape[1]:
                                st.write("Non-numeric columns detected in features. Converting...")
                                X = pd.get_dummies(X)

                            # Check and convert non-numeric target
                            if y.dtype == 'object':
                                st.write("Non-numeric target column detected. Converting...")
                                le = LabelEncoder()
                                y = le.fit_transform(y)

                            # Print shapes and types to diagnose issues
                            st.write("Features (X) shape:", X.shape)
                            st.write("Target (y) shape:", y.shape)
                            st.write("Features (X) types:\n", X.dtypes)
                            st.write("Target (y) type:", y.dtype)

                            # Ensure there are no missing values
                            if X.isnull().values.any() or pd.Series(y).isnull().values.any():
                                st.write("Error: Missing values detected. Please clean your data.")
                            else:
                                # Convert X to numpy array
                                X = X.values

                                # Random Forest parameters
                                n_estimators = st.slider("Number of Estimators", min_value=10, max_value=500, value=100, step=10)
                                max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=10, step=1)
                                min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=10, value=2, step=1)
                                criterion_classification = st.selectbox("Criterion for Classification", ["gini", "entropy"])
                                criterion_regression = st.selectbox("Criterion for Regression", ["squared_error", "absolute_error", "friedman_mse", "poisson"])

                                # Train-test split
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_split/100, random_state=42)

                                # Train Random Forest model
                                if len(set(y)) <= 2:  # Classification
                                    model = RandomForestClassifier(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        criterion=criterion_classification,
                                        random_state=42
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    accuracy = accuracy_score(y_test, y_pred)

                                    st.write("### Model Performance")
                                    st.write(f"Accuracy: {accuracy:.2f}")

                                else:  # Regression
                                    model = RandomForestRegressor(
                                        n_estimators=n_estimators,
                                        max_depth=max_depth,
                                        min_samples_split=min_samples_split,
                                        criterion=criterion_regression,
                                        random_state=42
                                    )
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    r2 = r2_score(y_test, y_pred)
                                    rmse = mean_squared_error(y_test, y_pred, squared=False)

                                    st.write("### Model Performance")
                                    st.write(f"R^2 Score: {r2:.2f}")
                                    st.write(f"RMSE: {rmse:.2f}")

                                # Display scatter plot of actual vs predicted values for regression
                                if len(set(y)) > 2:
                                    st.write("### Actual vs Predicted Values")
                                    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                                    st.line_chart(results_df)
                                model_path = 'Models/Randomforest_model.pkl'
                                # Button to save the model
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                if st.button('Save Random Forest Model'):
                                    with open(model_path, 'wb') as file:
                                        pickle.dump(model, file)

                                    st.write(f'Model will be saved to: {model_path}')
                                    st.success('Model saved successfully!')
            elif selected_tab==tabs[2]:
        # Title and description
                st.title('Train Test Split Data')

                # Sliders for Training Data Split and Testing Data Split
                training_split = st.slider('Training Data Split', 0, 100, 70)
                testing_split = 100 - training_split

                # Display the splits
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Training Data Split:')
                    st.write(training_split)
                with col2:
                    st.write('Testing Data Split:')
                    st.write(testing_split)

                # Split Data button
                if st.button('Split Data'):
                    st.write(f'Training Data: {training_split}%')
                    st.write(f'Testing Data: {testing_split}%')

        # Title and description
                st.title("Decision Tree Model")
                st.write("""
                This app allows you to upload a dataset (CSV or Excel), specify the target variable, and train a Decision Tree model.
                You can also evaluate the model's performance using Accuracy (for classification) and R^2 / RMSE (for regression).
                """)


                df = st.session_state['dataset']
                if df is not None:
                    st.write("Dataset Preview:")
                    st.write(df.head())

                    # Select target variable
                    target_column = st.selectbox("Select the target column", df.columns, key="target_column")

                    if target_column:
                        X = df.drop(columns=[target_column])
                        y = df[target_column]

                        # Check and convert non-numeric columns in features
                        if not X.select_dtypes(include=[float, int]).shape[1] == X.shape[1]:
                            st.write("Non-numeric columns detected in features. Converting...")
                            X = pd.get_dummies(X)

                        # Check and convert non-numeric target
                        if y.dtype == 'object':
                            st.write("Non-numeric target column detected. Converting...")
                            le = LabelEncoder()
                            y = le.fit_transform(y)

                        # Print shapes and types to diagnose issues
                        st.write("Features (X) shape:", X.shape)
                        st.write("Target (y) shape:", y.shape)
                        st.write("Features (X) types:\n", X.dtypes)
                        st.write("Target (y) type:", y.dtype)

                        # Ensure there are no missing values
                        if X.isnull().values.any() or pd.Series(y).isnull().values.any():
                            st.write("Error: Missing values detected. Please clean your data.")
                        else:
                            # Convert X to numpy array
                            X = X.values

                            # Choose model type
                            model_type = st.selectbox("Choose a model", ["Decision Tree Classifier", "Decision Tree Regressor"], key="model_type")

                            # Hyperparameters
                            if model_type == "Decision Tree Classifier":
                                criterion = st.selectbox("Criterion", ["gini", "entropy"], key="criterion")
                            else:  # Decision Tree Regressor
                                criterion = st.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"], key="criterion")
                                
                            max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=3, step=1, key="max_depth")
                            min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2, step=1, key="min_samples_split")

                            # Train-test split
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            # Train and evaluate the model
                            if model_type == "Decision Tree Classifier":
                                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                accuracy = accuracy_score(y_test, y_pred)

                                st.write("### Model Performance")
                                st.write(f"Accuracy: {accuracy:.2f}")

                            else:  # Decision Tree Regressor
                                model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                r2 = r2_score(y_test, y_pred)
                                rmse = mean_squared_error(y_test, y_pred, squared=False)

                                st.write("### Model Performance")
                                st.write(f"R^2 Score: {r2:.2f}")
                                st.write(f"RMSE: {rmse:.2f}")

                            # Display scatter plot of actual vs predicted values for regression
                            if model_type == "Decision Tree Regressor":
                                st.write("### Actual vs Predicted Values")
                                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                                st.line_chart(results_df)
                                model_path = 'Models/DecisionTree_model.pkl'
                                # Button to save the model
                                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                                if st.button('Save Decision Tree Model'):
                                    with open(model_path, 'wb') as file:
                                        pickle.dump(model, file)

                                    st.write(f'Model will be saved to: {model_path}')
                                    st.success('Model saved successfully!')


            elif selected_tab==tabs[3]:
        # Title and description
                st.title('Train Test Split Data')

                # Sliders for Training Data Split and Testing Data Split
                training_split = st.slider('Training Data Split', 0, 100, 70)
                testing_split = 100 - training_split

                # Display the splits
                col1, col2 = st.columns(2)
                with col1:
                    st.write('Training Data Split:')
                    st.write(training_split)
                with col2:
                    st.write('Testing Data Split:')
                    st.write(testing_split)

                # Split Data button
                if st.button('Split Data'):
                    st.write(f'Training Data: {training_split}%')
                    st.write(f'Testing Data: {testing_split}%')

                def load_data(uploaded_file):
                    if uploaded_file.name.endswith('.csv'):
                        return pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(uploaded_file)
                    else:
                        st.error("Unsupported file type")
                        return None

                def preprocess_data(df, target_column):
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

                    # Convert non-numeric columns in features
                    if not X.select_dtypes(include=[float, int]).shape[1] == X.shape[1]:
                        st.write("Non-numeric columns detected in features. Converting...")
                        X = pd.get_dummies(X)

                    # Convert non-numeric target
                    if y.dtype == 'object':
                        st.write("Non-numeric target column detected. Converting...")
                        le = LabelEncoder()
                        y = le.fit_transform(y)

                    # Ensure there are no missing values
                    if X.isnull().values.any() or pd.Series(y).isnull().values.any():
                        st.error("Missing values detected. Please clean your data.")
                        return None, None

                    return X.values, y

                def main():
                    st.title("Stacking Model")
                    st.write("""
                    This app allows you to upload a dataset (CSV or Excel), specify the target variable, and train a Stacking model.
                    You can also evaluate the model's performance using Accuracy for classification problems.
                    """)

                    uploaded_file = st.session_state['dataset']
                    if uploaded_file is not None:
                        try:
                            df = st.session_state['dataset']
                            if df is not None:
                                st.write("Dataset Preview:")
                                st.write(df.head())

                                target_column = st.selectbox("Select the target column", df.columns, key="target_column")

                                if target_column:
                                    X, y = preprocess_data(df, target_column)
                                    if X is not None and y is not None:
                                        st.write("Features (X) shape:", X.shape)
                                        st.write("Target (y) shape:", y.shape)

                                        # Check if dataset is large enough
                                        if len(y) < 10:
                                            st.error("Dataset too small. Please upload a dataset with at least 10 samples.")
                                            return

                                        rf_n_estimators = st.slider("Random Forest - Number of Estimators", min_value=10, max_value=500, value=100, step=10, key="rf_n_estimators")
                                        rf_max_depth = st.slider("Random Forest - Max Depth", min_value=1, max_value=10, value=3, step=1, key="rf_max_depth")
                                        
                                        gb_n_estimators = st.slider("Gradient Boosting - Number of Estimators", min_value=10, max_value=500, value=100, step=10, key="gb_n_estimators")
                                        gb_learning_rate = st.slider("Gradient Boosting - Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="gb_learning_rate")
                                        gb_max_depth = st.slider("Gradient Boosting - Max Depth", min_value=1, max_value=10, value=3, step=1, key="gb_max_depth")

                                        lr_C = st.slider("Logistic Regression - Regularization Strength (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, key="lr_C")

                                        test_size = st.slider("Test Size (proportion of data for testing)", min_value=0.1, max_value=0.5, value=0.2, step=0.01, key="test_size")

                                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                                        base_estimators = [
                                            ('rf', RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=42)),
                                            ('gb', GradientBoostingClassifier(n_estimators=gb_n_estimators, learning_rate=gb_learning_rate, max_depth=gb_max_depth, random_state=42))
                                        ]

                                        final_estimator = LogisticRegression(C=lr_C, random_state=42)

                                        model = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator)
                                        model.fit(X_train, y_train)

                                        y_pred = model.predict(X_test)
                                        accuracy = accuracy_score(y_test, y_pred)

                                        st.write("### Model Performance")
                                        st.write(f"Accuracy: {accuracy:.2f}")
                                        st.write("Classification Report:")
                                        st.text(classification_report(y_test, y_pred))
                        except Exception as e:
                            st.error(f"Error: {e}")

                main()

            # st.header("Auto Train ML models")
            
            # st.write("""
            # This section allows you to specify the target variable and train a Linear Regression model.
            # You can also evaluate the model's performance using R^2 and RMSE.
            # """)

            # if st.session_state['dataset'] is not None:
            #     df = st.session_state['dataset']
            #     st.write("Dataset Preview:")
            #     st.write(df.head())
                
            #     # Select target variable
            #     target_column = st.selectbox("Select the target column", df.columns)
            #     if target_column:
            #         X = df.drop(columns=[target_column])
            #         y = df[target_column]
                    
            #         # Check and convert non-numeric columns in features
            #         if not X.select_dtypes(include=[float, int]).shape[1] == X.shape[1]:
            #             st.write("Non-numeric columns detected in features. Converting...")
            #             X = pd.get_dummies(X)
                    
            #         # Check and convert non-numeric target
            #         if y.dtype == 'object':
            #             st.write("Non-numeric target column detected. Converting...")
            #             le = LabelEncoder()
            #             y = le.fit_transform(y)
                    
            #         # Print shapes and types to diagnose issues
            #         st.write("Features (X) shape:", X.shape)
            #         st.write("Target (y) shape:", y.shape)
            #         st.write("Features (X) types:\n", X.dtypes)
            #         st.write("Target (y) type:", y.dtype)
                    
            #         # Ensure there are no missing values
            #         if X.isnull().values.any() or pd.Series(y).isnull().values.any():
            #             st.write("Error: Missing values detected. Please clean your data.")
            #         else:
            #             # Convert X to numpy array
            #             X = X.values
                        
            #             # Train-test split
            #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
            #             # Train Linear Regression model
            #             model = LinearRegression()
            #             model.fit(X_train, y_train)
                        
            #             # Make predictions
            #             y_pred = model.predict(X_test)
                        
            #             # Calculate performance metrics
            #             r2 = r2_score(y_test, y_pred)
            #             rmse = mean_squared_error(y_test, y_pred, squared=False)
                        
            #             # Display performance metrics
            #             st.write("### Model Performance")
            #             st.write(f"R^2 Score: {r2:.2f}")
            #             st.write(f"RMSE: {rmse:.2f}")
                        
            #             # Display scatter plot of actual vs predicted values
            #             st.write("### Actual vs Predicted Values")
            #             results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            #             st.line_chart(results_df)

        elif selected_tab == "Freeze the learnings":
            st.title('Freeze the Learnings')

            # Header for the Linear Regression section
            st.header('Linear Regression')

            # Input for the model path
            model_path = st.text_input('Enter model path', 'Models/linear_model.pkl')

            # Button to save the model
            if st.button('Save Linear Regression Model'):
                st.write(f'Model will be saved to: {model_path}')
                st.success('Model saved successfully!')

        # Display footer
        st.write("Powered by Your Company")

    else:
        login_option = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
        if login_option == "Login":
            login_page(st.session_state['users'])
        else:
            register_page(st.session_state['users'])
home()
# Display the top header
