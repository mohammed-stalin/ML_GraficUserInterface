from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

def apply_dt_cm(df, target_column):
    try:
        # Assuming you have selected features (X) and target variable (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

        # Initialize the ID3 model
        dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

        # Fit the model
        dt_model.fit(X_train, y_train)

        # Make predictions
        y_pred = dt_model.predict(X_test)

        # Calculate confusion matrix and classification report
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        classification_report_result = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        # Display the results using the existing function
        return confusion_matrix_result, classification_report_result , accuracy
    except Exception as e:
        print(f"Error applying DT: {e}")
        return None
    