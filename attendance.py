import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title = "Attendance System",
    layout="wide",
    initial_sidebar_state="expanded"
    )

st.title("Face Recognition based Online Attendance System")
st.subheader("By Aditi Das & Aparna Sahu")

src_dir = os.path.join(os.getcwd(), "Attendance")
files = os.listdir(src_dir)
files = [filename.split('_')[1].replace('.csv', '') for filename in files]

st.sidebar.header("Options")
date_req = st.sidebar.selectbox("Select Attendance for Date: ", files)

df_name = f"{src_dir}/Attendance_{date_req}.csv"
df = pd.read_csv(df_name)


st.dataframe(df)