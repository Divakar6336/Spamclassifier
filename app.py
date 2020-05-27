#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from flask import render_template, url_for, request, Flask
import pickle


# In[8]:


clf=pickle.load(open('transform.pkl', 'rb'))
model=pickle.load(open('model.pkl', 'rb'))

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/Predict', methods=['POST'])
def predict():
    if request.method =='POST':
        message=request.form['message']
        data=[message]
        vect=clf.transform(data).toarray()
        my_prediction=model.predict(vect)
    return render_template('result.html', prediction=my_prediction)
        
        
        
        
if __name__=='__main__':
    app.run(debug=True)
        


# In[ ]:




