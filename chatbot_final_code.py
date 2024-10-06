import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Load intents file
data_file = open('intents3.json').read()
intents = json.loads(data_file)

# Tokenize patterns and categorize
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Output statistics
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to disk
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Initialize training data
train_x = []
train_y = []

# Create training data from documents
for doc in documents:
    bag = []
    output_row = [0] * len(classes)
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    
    output_row[classes.index(doc[1])] = 1
    
    train_x.append(bag)
    train_y.append(output_row)

# Convert training data to NumPy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model with SGD optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)
model.save('chatbot_model.h5')

print("Model created and saved")

# Load necessary files for chatbot response
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents3.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Clean up sentence by tokenizing and lemmatizing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convert sentence into bag-of-words
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

# Predict the intent
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

# Get a response based on intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Generate chatbot response
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# Example usage
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print(f"Bot: {chatbot_response(user_input)}")
