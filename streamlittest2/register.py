import streamlit as st
import hashlib

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, users):
    hashed_password = hash_password(password)
    users[username] = hashed_password

def register_page(users):
    st.title("Register")
    st.markdown("### Please fill in the details to create an account.")
    username = st.text_input("Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")
    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match")
        elif username in users:
            st.error("Username already exists")
        else:
            register_user(username, password, users)
            st.success("Registered successfully")