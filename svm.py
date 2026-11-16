from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
import matplotlib.pyplot as plt

def apply_svm_cm(df, target_column ):
    try:
        # Assuming you have selected features (X) and target variable (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

        # Create and train the SVM model
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_train, y_train)

        # Predictions on the test set
        y_pred = svm_model.predict(X_test)

        accuracy=accuracy_score(y_test,y_pred)
        # Evaluate the model
        confusion_matrix_result = confusion_matrix(y_test, y_pred)
        classification_report_result = classification_report(y_test, y_pred)

        # Display the results using the existing function
        return confusion_matrix_result, classification_report_result , accuracy
    except Exception as e:
        print(f"Error applying SVM: {e}")
        return None