import streamlit as st
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_credentials(username, password, users):
    hashed_password = hash_password(password)
    return users.get(username) == hashed_password

def login_page(users):
    st.title("Login")
    st.markdown("### Please enter your credentials to login.")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if check_credentials(username, password, users):
            st.session_state['logged_in'] = True
            st.success("Logged in successfully")
        else:
            st.error("Invalid username or password")