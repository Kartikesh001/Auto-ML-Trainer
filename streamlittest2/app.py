import streamlit as st
import pandas as pd

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

# Display the top header
st.markdown('<div class="top-header">AutoML</div>', unsafe_allow_html=True)

# Define the navigation buttons
tabs = ["Data Ingestion", "Data Transformation", "Auto Train ML models", "Freeze the learnings"]
selected_tab = st.radio("Navigation", tabs, index=0, horizontal=True, format_func=lambda x: f"{x}")


# Display content based on the selected tab
columnlist = None
if selected_tab == "Data Ingestion":
    st.header("Data Ingestion")
    
    # Choose data input method
    input_method = st.radio("Select data input method:", ["Enter file path", "Upload file"], horizontal=True)

    if input_method == "Enter file path":
        file_path = st.text_input("Path of the file", placeholder="Enter file path")
        file_name = st.text_input("Name of the file", placeholder="Enter file name with extension")
    elif input_method == "Upload file":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Ingest"):

            st.success("File ingested successfully!")
    with col2:
        st.write("")  # Placeholder for alignment
    
    

    if st.button("Run"):
        df = pd.read_excel(uploaded_file)
        columnlist = df.columns.tolist()
        st.subheader("Data dimensions:")
        cols = st.columns(2)
        with cols[0]:
            st.text_input("Number of rows", disabled=True, value=df.shape[0])
        with cols[1]:
            st.text_input("Number of columns", disabled=True, value=df.shape[1])
        st.success("Running data ingestion...")


elif selected_tab == "Data Transformation":
    # columns = df.columns.tolist()
    # options = ['Option 1', 'Option 2', 'Option 3', 'Option 4']
    selected_options = st.multiselect('Select one or more options:', columnlist)

    st.header("Data Transformation")
    st.write("Transformation options go here...")

elif selected_tab == "Auto Train ML models":
    st.header("Auto Train ML models")
    st.write("Model training options go here...")

elif selected_tab == "Freeze the learnings":
    st.header("Freeze the learnings")
    st.write("Saving and exporting model options go here...")

# Display footer
st.write("Powered by Your Company")