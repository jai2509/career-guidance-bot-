# <========================================================= Importing Required Libraries & Functions =================================================>
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
from PIL import Image
import random  # <--- Import the missing random module

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# <------------------------------------------------------------- Functions ----------------------------------------------------------------------------------->
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def chatbot_response(text):
    ints = predict_class(text, model)
    tag = ints[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])  # Uses random.choice to select a response

# <---------------------------------------------------------- Page Configuration -----------------------------------------------------------------------------> 
im = Image.open('bot.jpg')
st.set_page_config(layout="wide", page_title="Student's Career Counselling Chatbot", page_icon=im)

# <---------------------------------------------------------- Main Header ------------------------------------------------------------------------------------->
st.markdown(
    """
    <div style="background-color: #FF8C00 ; padding: 10px">
        <h1 style="color: brown; font-size: 48px; font-weight: bold">
           <center> <span style="color: black; font-size: 64px">C</span>areer <span style="color: black; font-size: 64px">B</span>uddy <span style="color: black; font-size: 64px">
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# <========================================================= Importing Data Files  ====================================================================>
with open('intents3.json', 'r') as file:
    intents = json.load(file)
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)
with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

# <--------------------------hide the right side streamlit menu button --------------------------------->
st.markdown(""" 
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style> 
""", unsafe_allow_html=True)

# <=========================================================== Sidebar ==============================================================================> 
with st.sidebar:
    st.title('''Personalized career assistance ''')
    
    st.markdown('''
    ## About~
    This app has been developed by 4 students of DTC :\n
    Jai Kumar Mishra [10918002721]\n
    Tushti Joshi [09918002721]\n
    Vidhush [07318002721]\n
    Sahil Aggarwal [10718002721]\n
    ''')
    add_vertical_space(5)
    
# <============================================================= Initializing Session State ==========================================================>
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm an AI Career Counselor, How may I help you?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

#<================================================== Function for taking user provided prompt as input ================================================>
def get_text():
    input_text = st.text_input("You: ", key="input", on_change=None)
    return input_text

styl = """
    <style>
        .stTextInput {
        position: fixed;
        bottom: 20px;
        z-index: 20;
        }
    </style>
"""
st.markdown(styl, unsafe_allow_html=True)

#<------------------------------------------------ Applying the user input box ------------------------------------------------------------------------>
with input_container:
    user_input = get_text()

# <================================================ Loading The Model ===============================================================>
model = load_model('chatbot_model.h5')

#<====================== Conditional display of AI generated responses as a function of user provided prompts =====================================>
with response_container:
    if user_input:
        response = chatbot_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state['generated'][i], key=str(i))

