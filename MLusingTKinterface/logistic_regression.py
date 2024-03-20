import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def apply_logistic_regression(df, target_column):
    try:
        # Assume the target column is binary (0 or 1)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

        # Initialize and fit the logistic regression model on the training set
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Calculate and print the accuracy score on the testing set
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"Model Accuracy: {accuracy}")
        # Display module score
        module_score = model.score(X_test, y_test)
        print(f"Model Score: {module_score}")

        # Calculate and print the confusion matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        print("Confusion Matrix:")
        print(cm)

        # Calculate and print the classification report
        classification_rep = classification_report(y_test, model.predict(X_test))
        print("Classification Report:")
        print(classification_rep)

        return cm, classification_rep ,accuracy

    except Exception as e:
        print(f"Error applying logistic regression: {e}")
        return None
