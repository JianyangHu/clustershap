from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_iris
import xgboost
import streamlit as st
import pandas as pd
from io import StringIO

from PIL import Image



st.title('Online ClusterShap')
st.title(':sunglasses:')


#########file upload##########

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write(data)



###########cluster##########

k=st.selectbox(
"cluster number",
set(range(0,10))
)

if st.button('Start clustershap'):
    
    km=KMeans(k).fit(data)

    #######xgb+shap####


    model=xgboost.XGBClassifier().fit(data,km.predict(data))


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data, show = False)
 
    plt.savefig("all.jpg")

    image1 = Image.open("all.jpg")
    
    st.image(image1)

    for i in range(k):
    

        plt.figure()
        shap.summary_plot(shap_values[i], data, show = False)

        plt.savefig(f"{i}.jpg")
        image1 = Image.open(f"{i}.jpg")
        st.image(image1)

        plt.show()


