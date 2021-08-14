import os
from datetime import datetime, date 
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("<h1 style='text-align: center; color: cornflowerblue;'>MedicAI 😷</h1>", unsafe_allow_html=True)

def homepage():
    # st.header('Welcome to MedicAI')
    st.markdown('<h1 style="text-align: center" style="font-size:180%"> Welcome to MedicAI </h1>', unsafe_allow_html=True)
    # st.markdown('<img src="C:/Users/Afrid/medicAI/Corona.jpg" class="centre" height = "200" width = "400">', unsafe_allow_html=True)
    st.image('./Corona.jpg')
    st.subheader('Coronaviruses are a large family of viruses that are known to cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS) and Severe Acute Respiratory Syndrome (SARS). A novel coronavirus (COVID-19) was identified in 2019 in Wuhan, China. Stay Home, Stay Safe.')

def cough_model():
    st.header("Cough Model 🤧")

    def feature_extraction(file_name):
        try:
            X, sr = librosa.load(file_name)
            if X.ndim > 1:
                X = X[:, 0]
            X = X.T
            stft = np.abs(librosa.stft(X))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            rmse = np.mean(librosa.feature.rms(y=X))
            spec_cent = np.mean(librosa.feature.spectral_centroid(y=X, sr=sr))
            spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=X, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=X, sr=sr))
            zcr = np.mean(librosa.feature.zero_crossing_rate(X))
            mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)  # 40 values
            return np.mean(chroma_stft), rmse, spec_cent, spec_bw, rolloff, zcr, mfcc 

        except ValueError:
            return None

    def model_r():
        col1, col2, col3 = st.columns(3)
        with col1:
            covid_stat = st.selectbox(
                "Severeness scale from 1-7 ",
                [1, 2, 3, 4, 5, 6, 7]
            )
            # st.write(f"Severeness Scale: {covid_stat}")
        with col2: 
            gender = st.selectbox(
                "Select Male or Female",
                ['Male', 'Female']
            )
            # st.write(f"You selected: {gender}")
        with col3: 
            age = st.number_input(
                'Enter your age', min_value=15, max_value=80, value=25, step=1
            )
            # st.write(f"Your age: {age}")

        filename = st.file_uploader('Upload Cough File in `.wav` format')

        if filename is not None:
            test = feature_extraction(filename)

        if st.button("Predict"):
            # preprocess data
            if gender == "Male":
                gender = 0
            else:
                gender = 1 
            data = pd.DataFrame([covid_stat, gender, age]).T
            features = pd.DataFrame(test).T
            featured_csv = pd.concat([data, features], axis=1)

            mfcc = pd.DataFrame()
            for i in range(len(featured_csv)):
                _, __ , Sxx = signal.spectrogram(featured_csv[6][i])
                a = pd.DataFrame(Sxx.T)
                mfcc = pd.concat([mfcc, a])

            df_ = pd.concat([featured_csv.drop([6], axis=1), mfcc], axis=1)
            model_pth = './cough_model/xgb.sav'
            loaded_model = pickle.load(open(model_pth, 'rb'))
            # loaded_model, df = model_r()
            result = loaded_model.predict(df_)
            if result==0:
                return st.success('Covid Negative')
            if result==1:
                return st.success('Covid Postivie')
    model_r()


def brain_model():
    st.header("Brain Model 🧠")
    upl = st.file_uploader('Upload Brain MRI file')
    if st.button("predict"):
        st.success('Covid Positive')

def lung_model():
    st.header("Lungs X-ray Model 🫁")
    
    test_dir = './lung_model/val_data/'
    DEVICE = 'cpu'

    def save_feature_vectors(model, loader, output_size=(1, 1), file="testb7"):
        model.eval()
        images = []
        for idx, (x, y) in enumerate(loader):
            x = x.to('cpu')
            with torch.no_grad():
                features = model.extract_features(x)
                features = F.adaptive_avg_pool2d(features, output_size=output_size)
            images.append(features.reshape(x.shape[0], -1).detach().cpu().numpy())
        X_val = np.concatenate(images, axis=0)
        return X_val

    def model_run(test_dir):
        # load efficientNet model
        model = EfficientNet.from_pretrained("efficientnet-b7")
        model._fc = nn.Linear(2560, 1)
        model.load_state_dict(torch.load('./lung_model/lung_model.pt', map_location=torch.device('cpu')))
        model.eval()

        # load the clf model 
        clf = pickle.load(open('./lung_model/lung_model.sav', 'rb'))

        # input transforms
        X_transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor()
        ])
        input_ = ImageFolder(test_dir, transform=X_transform)
        test_loader = DataLoader(input_)

        # save feature vectors as npy file   
        res = save_feature_vectors(model, test_loader, output_size=(1, 1), file="test_b7")
        return res, clf

    test_di = st.text_input('Enter path to x-ray images here')

    # classifier 
    if st.button("Prediction"):
        X_val, clf = model_run(test_dir)
        Result = clf.predict(X_val)
        for i in range(len(Result)):
            if Result[i]==0:
                print(st.success(f'Covid result: Negative'))
            else:
                print(st.success(f'Covid result: Positive'))


def vacc_forecast():
    st.header("Vaccination Drive Forecasting 📉")
    # df = pd.read_csv('./country_vaccinations.csv')
    df = pd.read_csv('./country_vaccinations_by_manufacturer.csv')
    df['date'] = pd.to_datetime(df['date'], format = '%Y/%m/%d')
    # To compelte the data, as naive method, we will use ffill
    f, ax = plt.subplots(nrows=len(df.columns)-1, ncols=1, figsize=(15, 25))

    for i, column in enumerate(df.drop('date', axis=1).columns):
        sns.lineplot(x=df['date'], y=df[column].fillna(method='ffill'), ax=ax[i], color='dodgerblue')
        ax[i].set_title('Feature: {}'.format(column), fontsize=18)
        ax[i].set_ylabel(ylabel=column, fontsize=18)
    st.pyplot(f)

opt = st.selectbox('',['Home', 'Cough', 'Lung', 'Brain',  'Forecast'])

if opt=='Home':
    homepage()
if opt=='Lung':
    lung_model()
if opt=='Cough':
    cough_model()
if opt=='Brain':
    brain_model()
if opt=='Forecast':
    vacc_forecast()



