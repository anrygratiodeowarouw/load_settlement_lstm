import streamlit as st
from engine import Regressor, device, predict_qi, plot_prediction
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Regressor(feature_size=4, hidden_size=[201], output_size=1, downscale=0, units=1, hidden_dim_nspt=3, out_feature_nspt=2).to(device)
model.load_state_dict(torch.load('model_col_2_1.018_3.671_2.199_14.05.pth', map_location=device))
model.eval()

st.title("Prediction of Load-settlement Curve of Pile Foundation using Deep Learning")
#membuat subheader
st.subheader("Thesis Project by: Anry Gratio Deo Warouw (25021069) Bandung Institute of Technology")
st.markdown("Selamat datang semua :partying_face:, saya Deo mahasiswa S2 Geoteknik sedang mengembangkan model AI :smile: dengan tujuan agar dapat digunakan untuk memprediksi Load-Settlement Tiang Bor di DKI Jakarta :satisfied:")

st.image('header.png')

col1, col2 = st.columns(2)

d = col1.selectbox('d', [0.8,1.0,1.2])
col = col2.number_input('col', 0.0)

col1, col2 = st.columns(2)
l = col1.number_input('l', 0.0)
n_ei = col2.slider('n_ei', 1, 25)

data_nspt = st.file_uploader('data nspt', ['csv'])


if data_nspt is not None:
    # Using Pandas to read the uploaded CSV file
    df = pd.read_csv(data_nspt)
    
    if st.button("Predict"):
        if l != 0 and n_ei != 1:
            result = []
            try:
                result = predict_qi(d, l, col, n_ei, df, model)
                st.write(result)
                st.pyplot(plot_prediction(n_ei, result))
            except ValueError:
                st.warning("Your NSPT CSV format is not correct")
                
        else:
            st.warning("Prediction can not be done because there's data that forbidden")
