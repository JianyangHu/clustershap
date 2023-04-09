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





    col = st.multiselect(
        'select columns',
        list(data.columns),
        list(data.columns))


    ###########cluster##########

    k=st.selectbox(
    "cluster number",
    set(range(0,10))
    )




if st.button('Start clustershap'):
    
    data=data[col]

    km=KMeans(k).fit(data)

    #####pca####
    pca=PCA().fit(data)
    embedding = pca.fit_transform(data)

    plt.figure()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=km.predict(data))
    plt.savefig("pca.jpg")

    image1 = Image.open("pca.jpg")
    
    st.image(image1)

    plt.show()



    #######xgb+shap####


    model=xgboost.XGBClassifier().fit(data,km.predict(data))


    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    plt.figure()
    shap.summary_plot(shap_values, data, show = False)
 
    plt.savefig("all.jpg")

    image1 = Image.open("all.jpg")
    
    st.image(image1)

    plt.show()

    for i in range(k):
    

        plt.figure()
        shap.summary_plot(shap_values[i], data, show = False)

        plt.savefig(f"{i}.jpg")
        image1 = Image.open(f"{i}.jpg")
        st.image(image1)

        plt.show()





