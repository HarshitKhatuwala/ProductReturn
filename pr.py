import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Function to preprocess the data
def preprocess_data(file_path):
    # Load dataset
    df = pd.read_excel(file_path)

    # Convert categorical variables to numerical using LabelEncoder
    encoder = LabelEncoder()
    df['category_encoded'] = encoder.fit_transform(df['category'])

    # Create feature matrix X and target vector y
    X = df[['customer_age', 'price', 'category_encoded']]
    y = df['return_status']  # Assuming 'return_status' is 1 for returned and 0 for not returned

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder

# Function to train the model
def train_model(X_train, y_train):
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    return accuracy

if __name__ == '__main__':
    # File path of the generated Excel dataset
    file_name = 'product_returns_sample_data.xlsx'
    file_path = os.path.join(os.path.dirname(__file__), file_name)

    # Preprocess data
    X_train, X_test, y_train, y_test, encoder = preprocess_data(file_path)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model and encoder
    model_path = 'product_return_prediction_model.pkl'
    encoder_path = 'category_encoder.pkl'
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    
    print(f'Model saved as {model_path} and encoder saved as {encoder_path}')
