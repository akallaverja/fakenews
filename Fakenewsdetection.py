import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem=PorterStemmer()
vect=TfidfVectorizer()

model=DecisionTreeClassifier()


vector_form=pickle.load(open('vector.pk1' , 'rb'))
load_model=pickle.load(open('model.pk1','rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ' , content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=vector_form.transform([news])
    prediction = load_model.predict(input_data)
    return prediction 

if __name__ == '__main__':
    st.title('Sistemi për zbulimin e lajmeve të rreme')
    st.subheader("Zbuloni tani nëse lajmi që po lexoni është real apo i rremë")
    sentence = st.text_area("Vendosni kontentin e lajmeve tuaja këtu", "Kontenti i lajmit", height=200)
    predict_btt = st.button("Kërko")
    
    if predict_btt:
        prediction_class = fake_news(sentence)
        if prediction_class == [0]:
            st.success('Lajmi eshte i vertete')
        elif prediction_class == [1]:
            st.warning('Lajmi eshte i rreme')