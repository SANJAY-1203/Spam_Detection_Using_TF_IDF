from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm, y_pred

def predict_email(model, vectorizer, text, clean_text_func):
    text = clean_text_func(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return "Spam" if prediction[0] == 1 else "Ham"
