import os
import shutil
import subprocess
import tempfile
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import webbrowser
from django import utils
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import customtkinter as ctk  # Import the CustomTkinter module
from sklearn.preprocessing import OneHotEncoder
from logistic_regression import *
from naive_bayes  import*
from kmeans import*
from svm import *
from id3 import *
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors, utils
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
import graphviz
import pandas as pd
from tabulate import tabulate
from io import StringIO
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

root_tk = tk.Tk()
root_tk.geometry("1500x800")
root_tk.title("ML")

class CustomApp:
    def __init__(self, master):
        self.master = master
        self.df = pd.DataFrame()  # Initialize an empty DataFrame
        
        # Attributes to store canvas and toolbar references
        self.tree = ttk.Treeview(self.master, columns=("Category", "Value"), show="headings")
        self.tree.heading("Category", text="Category")
        self.tree.heading("Value", text="Value")
        self.applied_cleanup_functions = []
        self.canvas = None
        self.toolbar = None
        
        # Create navigation bar on the left
        self.navigation_bar = CustomNavigationBar(self)
        self.navigation_bar.pack(side=tk.LEFT, fill=tk.Y)

    #mjid 
        self.classification_report_label = tk.Label(self.master, text="", font=('Times New Roman', 10), fg="#3498DB")
        self.classification_report_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
        # Table to display classification report
        self.classification_report_table = ttk.Treeview(self.master, show="headings", height=5)

        # Entry for the feature to one-hot encode
        self.encode_feature_label = tk.Label(self.master, text="Enter the feature to one-hot encode:")
        self.encode_feature_entry = tk.Entry(self.master)
        # Create the "Test2" button outside the navigation bar
        self.test2_btn = ctk.CTkButton(self.master, text="upload a file ", fg_color="#3498DB", command=self.upload_test2)
        self.test2_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=20,anchor=tk.CENTER)
        self.hide_test2_button()

        # Table to display data
        self.data_table = ttk.Treeview(self.master, show="headings", selectmode="browse", height=15)
        
        # Style for the Treeview
        style = ttk.Style()
        style.theme_use("clam")  # You can use "clam", "alt", or "default" based on your preference
        style.configure("Treeview", background="#ECF0F1", fieldbackground="#ECF0F1", foreground="#2C3E50", font=('Times New Roman', 9))
        style.map("Treeview", background=[("selected", "#3498DB")])

        # Table to display column information
        self.column_info_table = ttk.Treeview(self.master, show="headings", height=5)

        # Entry and Button for dropping a specific column
        self.column_label = tk.Label(self.master, text="Enter the column to drop ")
        self.column_entry = tk.Entry(self.master)
        self.drop_column_btn = ctk.CTkButton(self.master, text="Drop", fg_color="#3498DB", command=self.drop_column)
        
        # Entry for the feature to one-hot encode
        self.encode_feature_label = tk.Label(self.master, text="Enter the feature to one-hot encode:")
        self.encode_feature_entry = tk.Entry(self.master)
        self.encode_feature_btn = ctk.CTkButton(self.master, text="One-Hot Encode", fg_color="#3498DB", command=self.one_hot_encode)

        # Buttons for data cleanup
        self.cleanup_buttons = [
            ctk.CTkButton(self.master, text="Fill Missing with Mean", fg_color="#3498DB", command=lambda: self.handle_cleanup_action("mean")),
            ctk.CTkButton(self.master, text="Fill Missing with Median", fg_color="#3498DB", command=lambda: self.handle_cleanup_action("median")),
            ctk.CTkButton(self.master, text="Drop Rows with Missing Values", fg_color="#3498DB", command=lambda: self.handle_cleanup_action("drop")),
        ]
        # Buttons for visualization
        self.visualization_buttons = [
            ttk.Button(self.master, text="Histogram", command=lambda: self.handle_visualization_action("Histogram")),
            ttk.Button(self.master, text="Heatmap", command=lambda: self.handle_visualization_action("Heatmap")),
            ttk.Button(self.master, text="Pointplot", command=lambda: self.handle_visualization_action("Pointplot")),
            ttk.Button(self.master, text="Box Plot", command=lambda: self.handle_visualization_action("Box Plot")),
        ]

        self.classification_button = None  #Initialize the button reference

        # Entry and Button for the target column
        self.target_label = tk.Label(self.master, text="Enter your target ")
        self.numberofclusters_label = tk.Label(self.master, text="Enter number of clusters ")
        self.target_entry = tk.Entry(self.master)
        self.validate_target_btn = ctk.CTkButton(self.master, text="Validate", fg_color="#3498DB", command=self.validate_classification)
        # Assuming you have a list of classification algorithms
        self.algorithm_options = ["Naive Bayes","Logistic Regression","SVM","DT"]  # Update with your algorithm names
        self.algorithm_var = tk.StringVar(value=self.algorithm_options[0])
        self.algorithm_dropdown = tk.OptionMenu(self.master, self.algorithm_var, *self.algorithm_options)

        self.cluster_entry = tk.Entry(self.master)
        self.validate_unsup_btn = ctk.CTkButton(self.master, text="Validate", fg_color="#3498DB", command=self.unsup_classification)
        # Assuming you have a list of classification algorithms
        self.unsup_algorithm_options = ["Kmeans"]  # Update with your algorithm names
        self.unsup_algorithm_var = tk.StringVar(value=self.unsup_algorithm_options[0])
        self.unsup_algorithm_dropdown = tk.OptionMenu(self.master, self.unsup_algorithm_var, *self.unsup_algorithm_options)

        # New label to display results
        self.results_label = tk.Label(self.master, text="", font=('Times New Roman', 12), fg="#3498DB")
        self.results_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
        # Label wah sup wla unsuper
        self.learning_type_var = tk.StringVar(value="Type d'Apprentissage")  # Set default value
        self.learning_type_dropdown = tk.OptionMenu(self.master, self.learning_type_var, "supervised", "unsupervised", command=self.on_learning_type_change)
        self.learning_type_dropdown.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
        self.hide_type_classification()

        # Welcome Message
        # Load the welcome image
        self.welcome_image = Image.open("home1.png")
        self.welcome_photo = ImageTk.PhotoImage(self.welcome_image)
        # Create a label to display the welcome image
        self.welcome_label = tk.Label(self.master, text="WELCOME TO YOUR MACHINE LEARNING APPLICATION", font=('Times New Roman', 20), fg="#3498DB")
        self.welcome_label.pack(side=tk.TOP, padx=40, pady=20)
        self.welcome_image = tk.Label(self.master, image=self.welcome_photo)
        self.welcome_image.pack( side=tk.TOP, fill=tk.X,padx=00, pady=20)
        self.description_label = tk.Label(self.master, text="the application is a versatile machine learning tool with a focus on user experience, allowing you to interact with\n machine learning models through this graphical interface for tasks such as uploading data, selecting models,\nand evaluating their performance.", font=('Times New Roman', 16), fg="#000000")
        self.description_label.pack(anchor=tk.W,fill=tk.X, padx=40, pady=20)
        self.prepared_label = tk.Label(self.master, text="Developped by :", font=('Times New Roman', 16), fg="#3498DB")
        self.prepared_label.pack(anchor=tk.W, padx=40, pady=0)
        self.devolopers_name = tk.Label(self.master, text="Mohammed AACHABI & Abdelmajid BENJELLOUN", font=('Times New Roman', 14), fg="#000000")
        self.devolopers_name.pack(anchor=tk.W, padx=40, pady=0)
        self.encadree_label = tk.Label(self.master, text="Framed by :", font=('Times New Roman', 16), fg="#3498DB")
        self.encadree_label.pack(anchor=tk.W, padx=40, pady=0)
        self.prof_name = tk.Label(self.master, text="Prof Mme Sanae KHALI ISSA ", font=('Times New Roman', 14), fg="#000000")
        self.prof_name.pack(anchor=tk.W, padx=40, pady=0)
        # Add a button for displaying the manual in the home section
        self.manual_button = ctk.CTkButton(self.master, text="User Manual", fg_color="#3498DB", command=self.display_manual)
        self.manual_button.pack(anchor=tk.E, padx=70, pady=0)
        self.warning_name = tk.Label(self.master, text="Our project can make mistakes. Considier checking important information ", font=('Times New Roman', 8), fg="#333333")
        self.warning_name.pack(anchor=tk.E, padx=40, pady=0)



        self.accuracy_label = tk.Label(self.master, text="", font=('Times New Roman', 16), fg="#000000")



    def show_home_components(self):
        self.welcome_label.pack(side=tk.TOP, padx=40, pady=20)
        self.welcome_image.pack(side=tk.TOP, fill=tk.X, padx=00, pady=20)
        self.description_label.pack(anchor=tk.W ,fill=tk.X,padx=40, pady=20)
        self.prepared_label.pack(anchor=tk.W, padx=40, pady=0)
        self.devolopers_name.pack(anchor=tk.W, padx=40, pady=0)
        self.encadree_label.pack(anchor=tk.W, padx=40, pady=0)
        self.prof_name.pack(anchor=tk.W, padx=40, pady=0)
        self.manual_button.pack(anchor=tk.E,  padx=70, pady=0)
        self.warning_name.pack(anchor=tk.E, padx=40, pady=0)

    def hide_home_components(self):
        # Hide the components in the "Home" section
        self.welcome_image.pack_forget()
        self.welcome_label.pack_forget()
        self.prepared_label.pack_forget()
        self.devolopers_name.pack_forget()
        self.description_label.pack_forget()
        self.encadree_label.pack_forget()
        self.prof_name.pack_forget()
        self.warning_name.pack_forget()
        self.manual_button.pack_forget()
        
    
    def display_manual(self):
        # Get the current directory
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Construct the path to the manual PDF file
        manual_path = os.path.join(current_directory, 'Manuel_of_use.pdf')

        # Check if the manual PDF file exists
        if os.path.exists(manual_path):
            # Open the manual PDF file in the default PDF viewer
            webbrowser.open(manual_path)
        else:
            # Display an error message if the manual PDF file is not found
            tk.messagebox.showerror("Error", "Manual not found: manual.pdf")


    def export_to_pdf(self):
        if not self.df.empty:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])

            if file_path:
                try:
                    doc = SimpleDocTemplate(file_path, pagesize=letter)
                    elements = []

                    # Add content to the PDF
                    elements.append(self.create_table(file_path))

                    doc.build(elements)

                    print(f"PDF report saved to: {file_path}")

                except Exception as e:
                    print(f"Error exporting to PDF: {e}")

    def create_table(self, file_path):
        # Extract data for the table
        first_last_values = self.get_first_last_values()
        column_data = self.get_column_data()
        describe_data = self.get_describe_data()

        # Cleanup functions
        cleanup_functions = []
        if "mean" in self.applied_cleanup_functions:
            cleanup_functions.append("1. Fill Missing with Mean")
        if "median" in self.applied_cleanup_functions:
            cleanup_functions.append("2. Fill Missing with Median")
        if "drop" in self.applied_cleanup_functions:
            cleanup_functions.append("3. Drop Rows with Missing Values")
        if "one_hot_encode" in self.applied_cleanup_functions:
            cleanup_functions.append("4. One-Hot Encode")

        # Combine all the data into a structured format
        data = [
            ["Report of your file"],
            ["File Path:", file_path],
            ["-" * 50],  # Line separator
            ["First and Last Values"],
            *first_last_values,
            ["-" * 50],  # Line separator
            ["Column Name", "Data Type"],
            *column_data,
            ["-" * 50],  # Line separator
            ["Describe Output"],
            *describe_data,
            ["-" * 50],  # Line separator
            ["Cleanup Functions Used"],
            *cleanup_functions
        ]

        # Define styles for the table
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            # ('FONTNAME', (0, 0), (-1, 0), 'Helvitica'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ])

        # Create the table
        table = Table(data, style=style)

        return table


    def get_first_last_values(self):
    # Extract first and last values of each column
        first_last_values = []
        for col in self.df.columns:
            first_value = self.df[col].iloc[0]
            last_value = self.df[col].iloc[-1]
            first_last_values.append([f"{col} - First Value:", first_value])
            first_last_values.append([f"{col} - Last Value:", last_value])
        return first_last_values

    def get_column_data(self):
        # Extract column names and data types
        column_data = [[col, str(dtype)] for col, dtype in zip(self.df.columns, self.df.dtypes)]
        return column_data

    def get_describe_data(self):
        # Extract statistics from file.describe()
        describe_data = [["Statistics"]] + list(zip(self.df.describe().index, self.df.describe().values))
        return describe_data
    



    #fonction bach nchufo wach sup wla unsup 
    def show_type_classification(self):
        if not self.df.empty:
            # self.learning_type_var.set("supervised")  # Set default value
            self.learning_type_dropdown.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def hide_type_classification(self):
        self.learning_type_dropdown.pack_forget()

    def validate_classification(self):
        target_column = self.target_entry.get()
        algorithm = self.algorithm_var.get()
        if target_column in self.df.columns:
            print(f"Target column: {target_column}")
            print(f"Selected algorithm: {algorithm}")

            # Check the selected algorithm and call the corresponding code
            if algorithm == "Logistic Regression":
                confusion_matrix_result ,classification_report_result , accuracy = apply_logistic_regression(self.df, target_column)
                
                if hasattr(self, 'results_label'):
                    self.results_label.pack_forget()

                # Check if canvas exists and hide it
                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()

                # Check if toolbar exists and hide it
                if self.toolbar:
                    self.toolbar.pack_forget()

            elif algorithm == "Naive Bayes":
                confusion_matrix_result, classification_report_result ,accuracy = apply_naive_bayes(self.df, target_column)
                if hasattr(self, 'results_label'):
                    self.results_label.pack_forget()

                # Check if canvas exists and hide it
                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()

                # Check if toolbar exists and hide it
                if self.toolbar:
                    self.toolbar.pack_forget()
            elif algorithm == "SVM":
                self.apply_svm()
                confusion_matrix_result , classification_report_result , accuracy = apply_svm_cm(self.df, target_column )
            elif algorithm == "DT":
                if hasattr(self, 'results_label'):
                    self.results_label.pack_forget()

                # Check if canvas exists and hide it
                if self.canvas:
                    self.canvas.get_tk_widget().pack_forget()

                # Check if toolbar exists and hide it
                if self.toolbar:
                    self.toolbar.pack_forget()
                self.apply_dt()
                confusion_matrix_result , classification_report_result , accuracy = apply_dt_cm(self.df, target_column )
        # Add other algorithms as needed
            if confusion_matrix_result is not None:
                print("Confusion Matrix:")
                print(confusion_matrix_result)
                print(classification_report_result)
                print(accuracy)
                
                # Display the results using the new function
                results_text = f"Confusion Matrix:\n{confusion_matrix_result}\n\nClassification Report:\n{classification_report_result}"
                self.display_results(confusion_matrix_result,classification_report_result,accuracy)
        else:
            print(f"Column not found: {target_column}")


    def unsup_classification(self):
        algorithm = self.unsup_algorithm_var.get()
        print(f"Selected algorithm: {algorithm}")

        # Check the selected algorithm and call the corresponding code
        if algorithm == "Kmeans":
            self.apply_kmeans()

    def apply_kmeans(self):
        cluster=int(self.cluster_entry.get())
        try:
            # Assuming df is the DataFrame with the loaded data
            if not self.df.empty:
                # Select only numeric columns for clustering
                numeric_cols = self.df.select_dtypes(include=['number'])
                # Apply KMeans algorithm
                kmeans = KMeans(n_clusters=cluster, n_init=10)  # Explicitly set n_init to suppress warning
                self.df['Cluster'] = kmeans.fit_predict(numeric_cols)

                # Visualize the results using PCA for dimensionality reduction
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(numeric_cols)

                # Create a scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.df['Cluster'], cmap='viridis')
                plt.title('KMeans Clustering Results')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')

                # Embed the plot in Tkinter window
                self.embed_plot(plt)

        except Exception as e:
            print(f"Error applying KMeans: {e}")

    def apply_svm(self):
        target_column = self.target_entry.get()
        try:
            # Assuming df is the DataFrame with the loaded data
            if not self.df.empty and target_column in self.df.columns:
                # Split the data into features and target
                X = self.df.drop(target_column, axis=1)
                y = self.df[target_column]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

                # Initialize the SVM model
                svm_model = SVC()

                # Fit the model
                svm_model.fit(X_train, y_train)

                # Make predictions
                y_pred = svm_model.predict(X_test)
                # Visualize the results using PCA for dimensionality reduction
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(X_test)

                # Create a scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred, cmap='viridis')
                plt.title('SVM Classification Results')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')

                # Embed the plot in Tkinter window
                self.embed_plot(plt)

            else:
                print("Data or target column not available for SVM classification.")

        except Exception as e:
            print(f"Error applying SVM: {e}")


    def apply_dt(self):
        target_column = self.target_entry.get()
        try:
            # Assuming df is the DataFrame with the loaded data
            if not self.df.empty and target_column in self.df.columns:
                # Split the data into features and target
                X = self.df.drop(target_column, axis=1)
                y = self.df[target_column]

                # Initialize the Decision Tree model
                dt_model = DecisionTreeClassifier()

                # Fit the model
                dt_model.fit(X, y)

                # Visualize the decision tree
                dot_data = export_text(dt_model, feature_names=X.columns.tolist(), spacing=3)
                # Replace special characters (e.g., '|') with something else (e.g., '_')
                dot_data = dot_data.replace('|', '_')

                # Create a graphical representation of the decision tree
                graph_data = export_graphviz(dt_model, feature_names=X.columns.tolist(), class_names=list(map(str, dt_model.classes_)), filled=True, rounded=True, special_characters=True)

                # Replace special characters in the graph data
                graph_data = graph_data.replace('|', '_')

                # Save the modified graph data to a temporary file
                tmp_dot_filename = tempfile.mktemp(suffix=".dot")
                with open(tmp_dot_filename, 'w') as tmp_dot_file:
                    tmp_dot_file.write(graph_data)

                # Convert the DOT file to PNG using Graphviz
                png_filename = tempfile.mktemp(suffix=".png")
                subprocess.run(["dot", "-Tpng", "-o", png_filename, tmp_dot_filename], check=True)

                # Display the image in a new window
                self.display_image_in_window(png_filename)

        except Exception as e:
            print(f"Error applying Decision Tree: {e}")

    def display_image_in_window(self, image_path):
        # Create a new Toplevel window
        image_window = tk.Toplevel(self.master)

        # Load the image
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        # Create a label to display the image
        image_label = tk.Label(image_window, image=photo)
        image_label.photo = photo
        image_label.pack()

        # Automatically save the image to a predefined location
        self.save_image(image_path)

    def save_image(self, image_path):
        # Specify the predefined destination path
        destination_path = "tree.png"

        # Copy the image to the destination path
        shutil.copyfile(image_path, destination_path)

        print(f"Image saved to: {destination_path}")


    def embed_plot(self, plt):
        try:
            # Check if canvas already exists and destroy it to avoid duplication
            if self.canvas:
                self.canvas.get_tk_widget().destroy()

            # Create a Tkinter canvas to embed the matplotlib plot
            self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.master)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Check if toolbar already exists and destroy it to avoid duplication
            if self.toolbar:
                self.toolbar.destroy()

            # Create a toolbar for the plot
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        except Exception as e:
            print(f"Error embedding plot: {e}")


    def on_learning_type_change(self, *args):
        # Method to be called when the learning type dropdown value changes
        learning_type = self.learning_type_var.get()

        if learning_type == "supervised":
            self.show_validate_classification()
            self.hide_unsup_classification()
        elif learning_type == "unsupervised":
            self.hide_accuracy_label()
            self.hide_validate_classification()
            self.show_unsup_classification()
        else:
            self.hide_validate_classification()
            self.hide_unsup_classification()

    def display_results(self, confusion_matrix_result, classification_report_result,accuracy):
        results_text = f"Confusion Matrix:\n{confusion_matrix_result}\n\nClassification Report:\n{classification_report_result}"
        accuracy_text = f"Accuracy:\n{accuracy}"
        self.results_label.config(text=results_text)
        self.accuracy_label.config(text=accuracy_text)
        self.results_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
        self.accuracy_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def hide_accuracy_label(self):
        self.accuracy_label.pack_forget()

    def hide_validate_classification(self):
        if not self.df.empty:
            self.target_entry.pack_forget()
            self.target_label.pack_forget()
            self.validate_target_btn.pack_forget()
            self.algorithm_dropdown.pack_forget()
            self.results_label.pack_forget()
            # Hide the results_label only if it exists
            if hasattr(self, 'results_label'):
                self.results_label.pack_forget()

            # Check if canvas exists and hide it
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()

            # Check if toolbar exists and hide it
            if self.toolbar:
                self.toolbar.pack_forget()

    def show_validate_classification(self):
        if not self.df.empty:
            self.algorithm_dropdown.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            self.target_entry.delete(0, tk.END)  # Clear any previous text
            self.target_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=5)
            self.target_entry.pack(side=tk.TOP, fill=tk.X, padx=40, pady=5)
            self.validate_target_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
    
    def hide_unsup_classification(self):
        if not self.df.empty:
            self.validate_unsup_btn.pack_forget()
            self.numberofclusters_label.pack_forget()
            self.cluster_entry.pack_forget()
            self.unsup_algorithm_dropdown.pack_forget()
            # Hide the results_label only if it exists
            if hasattr(self, 'results_label'):
                self.results_label.pack_forget()

            # Check if canvas exists and hide it
            if self.canvas:
                self.canvas.get_tk_widget().pack_forget()

            # Check if toolbar exists and hide it
            if self.toolbar:
                self.toolbar.pack_forget()

    def show_unsup_classification(self):
        if not self.df.empty:
            self.unsup_algorithm_dropdown.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            self.cluster_entry.delete(0, tk.END)  # Clear any previous text
            
            self.numberofclusters_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=5)
            self.cluster_entry.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            self.validate_unsup_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def create_gui(self):
        # Create your main GUI elements here
        self.tree = ttk.Treeview()  # You may need to configure the treeview according to your needs
        self.tree["columns"] = ("Metric", "Value")
        self.tree.heading("#0", text="Category")
        self.tree.heading("Metric", text="Metric")
        self.tree.heading("Value", text="Value")

    def hide_test2_button(self):
        self.test2_btn.pack_forget()

    def show_test2_button(self):
        self.test2_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=20)  # Adjusted packing parameters
        self.test2_btn.focus_set()  # Set focus on the button

    
    def show_classification_button(self):
        self.classification_button = ctk.CTkButton(self.master, text="Classification", fg_color="#3498DB", command=lambda: self.on_button_click("Classification"))
        self.classification_button.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def hide_classification_button(self):
        if self.classification_button:
            self.classification_button.pack_forget()


    def upload_test2(self):
        file_path = filedialog.askopenfilename(title="Select a file", filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*")])
        if file_path:
            try:
                # Check the file extension
                _, file_extension = os.path.splitext(file_path)

                if file_extension.lower() == ".csv":
                    # Read data from the CSV file
                    self.df = pd.read_csv(file_path)
                elif file_extension.lower() == ".xlsx" or file_extension.lower() == ".xls" :
                    # Read data from the Excel file
                    self.df = pd.read_excel(file_path)
                else:
                    print("Unsupported file format. Please select a CSV or Excel file.")
                    return

                # Display data in the main table
                self.display_data(self.df)

                # Display column information in a separate table
                self.display_column_info(self.df)

            except Exception as e:
                print(f"Error loading file: {e}")


    def display_data(self, df):
        # Clear previous data in the main table
        self.data_table.delete(*self.data_table.get_children())

        # Update columns in the main table
        self.data_table["columns"] = tuple(df.columns)
        for col in df.columns:
            self.data_table.heading(col, text=col)
            self.data_table.column(col, anchor=tk.CENTER)

        # Insert data into the main table
        for i, row in df.iterrows():
            self.data_table.insert("", i, values=tuple(row))

        # Show the main table below the "Test2" button
        self.data_table.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def display_column_info(self, df):
        # Clear previous data in the column information table
        self.column_info_table.destroy()

        # Create a new column information table
        self.column_info_table = ttk.Treeview(self.master, show="headings", height=5)

        # Style for the Treeview (same style as the main table)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background="#ECF0F1", fieldbackground="#ECF0F1", foreground="#2C3E50", font=('Times New Roman', 9))
        style.map("Treeview", background=[("selected", "#3498DB")])

        # Display column information in the new table
        self.column_info_table["columns"] = ("Name", "Data Type", "Shape")
        self.column_info_table.heading("Name", text="Name")
        self.column_info_table.heading("Data Type", text="Data Type")
        self.column_info_table.heading("Shape", text="Shape")

        for col in df.columns:
            col_info = (col, str(df[col].dtype), str(df[col].shape))
            self.column_info_table.insert("", "end", values=col_info)

        # Show the column information table below the main table
        self.column_info_table.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def hide_cleanup_buttons(self):
        if not self.df.empty:
            for button in self.cleanup_buttons:
                button.pack_forget()

    def show_cleanup_buttons(self):
        if not self.df.empty:
            for button in self.cleanup_buttons:
                button.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def handle_cleanup_action(self, action):
        success = False
        # Assuming df is the DataFrame with the loaded data
        if action == "mean":
            try:
                self.df.fillna(self.df.mean(), inplace=True)
                print("Filled missing values with mean.")
                success = True
            except:
                print("theres a categorical values try one hot encoder before.")
        elif action == "median":
            try:
                self.df.fillna(self.df.median(), inplace=True)
                print("Filled missing values with median.")
                success = True
            except:
                print("theres a categorical values try one hot encoder before.")
        elif action == "drop":
            self.df.dropna(inplace=True)
            print("Dropped rows with missing values.")


        self.display_data(self.df)
        
        if success:
            self.hide_cleanup_buttons()
            self.hide_onehot_column_section()
            self.hide_drop_column_section()
            # Show the "Go to Classification" button after cleanup
            self.show_classification_button()

    def drop_column(self):
        column_name = self.column_entry.get()
        if column_name in self.df.columns:
            self.df.drop(column_name, axis=1, inplace=True)
            print(f"Dropped column: {column_name}")
            self.display_data(self.df)
            self.column_entry.delete(0, tk.END)
        else:
            print(f"Column not found: {column_name}")

    def one_hot_encode(self):
        # Get the feature name from the entry
        feature_to_encode = self.encode_feature_entry.get()

        # Assuming df is the DataFrame with the loaded data
        if not self.df.empty and feature_to_encode in self.df.columns:
            # Perform one-hot encoding for the selected feature
            one_hot_encoded = pd.get_dummies(self.df[feature_to_encode], prefix=feature_to_encode, dummy_na=False)

            # Concatenate the one-hot-encoded DataFrame with the original DataFrame
            self.df = pd.concat([self.df, one_hot_encoded], axis=1)

            # Drop the original categorical column
            self.df.drop(feature_to_encode, axis=1, inplace=True)
            print(f"One-Hot Encoding completed for feature: {feature_to_encode}. Original feature dropped.")
            self.encode_feature_entry.delete(0, tk.END)
            # Display the updated data
            self.display_data(self.df)
        elif feature_to_encode not in self.df.columns:
            print(f"Feature not found: {feature_to_encode}")
        else:
            print("No data to one-hot encode.")


    def hide_data_table(self):
        self.data_table.pack_forget()
        self.show_visualization_buttons()

    def hide_visualization_buttons(self):
        for button in self.visualization_buttons:
            button.pack_forget()

    def show_visualization_buttons(self):
        if not self.df.empty:
            for button in self.visualization_buttons:
                button.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def handle_visualization_action(self, action):
        try:
            if action == "Histogram":
                if not self.df.empty:
                    numeric_cols = self.df.select_dtypes(include=['number'])

                    # Create a single figure
                    plt.figure(figsize=(12, 8))

                    # Loop through numeric columns and create histograms in subplots
                    for i, col in enumerate(numeric_cols.columns, 1):
                        plt.subplot(1, len(numeric_cols.columns), i)
                        sns.histplot(self.df[col], bins=20, kde=True)
                        plt.title(f'Histogram for {col}')
                        plt.xlabel(col)
                        plt.ylabel('Frequency')

                    # Adjust layout for better spacing
                    plt.tight_layout()

                    # Show the combined plot
                    plt.show()
                else:
                    print("DataFrame is empty. Unable to plot histograms.")
            elif action == "Heatmap":
                plt.figure(figsize=(10, 8))
                sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Correlation Heatmap')
                plt.show()

            elif action == "Pointplot":
                if len(self.df.columns) >= 2:
                    plt.figure(figsize=(8, 6))
                    sns.pointplot(x=self.df.columns[0], y=self.df.columns[1], data=self.df)
                    plt.title('Point Plot')
                    plt.xlabel(self.df.columns[0])
                    plt.ylabel(self.df.columns[1])
                    plt.show()

            elif action == "Box Plot":
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if not numeric_cols.empty:
                    # Create a single figure
                    plt.figure(figsize=(12, 8))

                    # Loop through numeric columns and create box plots in subplots
                    for i, col in enumerate(numeric_cols, 1):
                        plt.subplot(1, len(numeric_cols), i)
                        sns.boxplot(x=self.df[col])
                        plt.title(f'Box Plot for {col}')
                        plt.xlabel(col)
                        plt.ylabel('Values')

                    # Adjust layout for better spacing
                    plt.tight_layout()

                    # Show the combined plot
                    plt.show()
                else:
                    print("No numeric columns in the DataFrame. Unable to plot box plots.")

            else:
                print(f"Unknown action: {action}")
        except Exception as e:
            print(f"Error performing {action}action:{e}")


    def on_button_click(self, text):
        if text == "Home":
            self.show_home_components()
            self.hide_data_table()
            self.hide_type_classification()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_test2_button()
            self.hide_classification_button()
            self.hide_classification_button()
            self.hide_validate_classification()
            self.hide_unsup_classification()
            self.hide_visualization_buttons()
            self.hide_onehot_column_section()
            self.hide_accuracy_label()

        elif text == "Upload":
            self.hide_onehot_column_section()
            self.hide_home_components()
            self.show_test2_button()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_classification_button()
            self.hide_validate_classification()
            self.hide_data_table()
            self.hide_type_classification()
            self.hide_unsup_classification()
            self.hide_visualization_buttons()
            self.hide_accuracy_label()

        elif text == "Data Cleanup":
            self.hide_home_components()
            self.show_data_table()
            self.hide_test2_button()
            self.hide_column_info_table()
            self.hide_validate_classification()
            self.show_cleanup_buttons()
            self.show_onehot_column_section()
            self.show_drop_column_section()
            self.hide_type_classification()
            self.hide_unsup_classification()
            self.hide_visualization_buttons()
            self.hide_accuracy_label()

        elif text == "Classification":
            self.hide_home_components()
            self.show_type_classification()
            self.show_validate_classification()
            self.hide_test2_button()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_classification_button()
            self.hide_data_table()
            self.hide_visualization_buttons()
            self.hide_onehot_column_section()

        elif text == "Export":
            self.hide_home_components()
            self.export_to_pdf()
            self.show_data_table()
            self.hide_type_classification()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_test2_button()
            self.hide_classification_button()
            self.hide_classification_button()
            self.hide_validate_classification()
            self.hide_unsup_classification()
            self.hide_visualization_buttons() 
            self.hide_onehot_column_section() 
            self.hide_accuracy_label()          

        elif text == "Visualization":
            self.hide_home_components()
            self.show_data_table()
            self.hide_type_classification()
            self.hide_test2_button()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_classification_button()
            self.hide_validate_classification()
            self.hide_unsup_classification()
            self.visualize_data()
            self.hide_onehot_column_section()
            self.hide_accuracy_label()


        else:
            self.hide_accuracy_label()
            self.hide_home_components()
            self.hide_data_table()
            self.hide_type_classification()
            self.hide_test2_button()
            self.hide_column_info_table()
            self.hide_cleanup_buttons()
            self.hide_drop_column_section()
            self.hide_classification_button()
            self.hide_validate_classification()
            self.hide_data_table()
            self.hide_unsup_classification()
            self.show_visualization_buttons()
            self.hide_onehot_column_section()


    def hide_drop_column_section(self):
        if not self.df.empty:
            self.column_label.pack_forget()
            self.column_entry.pack_forget()
            self.drop_column_btn.pack_forget()

    def show_drop_column_section(self):
        if not self.df.empty:
            self.column_entry.delete(0, tk.END)  # Clear any previous text
            self.column_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            self.column_entry.pack(side=tk.TOP, fill=tk.X, padx=40, pady=5)
            self.drop_column_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
    
    def hide_onehot_column_section(self):
        if not self.df.empty:
            self.encode_feature_label.pack_forget()
            self.encode_feature_entry.pack_forget()
            self.encode_feature_btn.pack_forget()

    def show_onehot_column_section(self):
        if not self.df.empty:
            self.encode_feature_entry.delete(0, tk.END)  # Clear any previous text
            self.encode_feature_label.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            self.encode_feature_entry.pack(side=tk.TOP, fill=tk.X, padx=40, pady=5)
            self.encode_feature_btn.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)

    def hide_column_info_table(self):
        self.column_info_table.pack_forget()

    def show_data_table(self):
        if not self.df.empty:
            self.data_table.pack(side=tk.TOP, fill=tk.X, padx=40, pady=10)
            if hasattr(self, 'visualization_pressed') and self.visualization_pressed:
                self.visualization_pressed = False  # Reset the flag
                self.visualize_data()  # Call visualize_data if the button was pressed

class CustomNavigationBar(tk.Frame):
    def __init__(self, app, master=None):
        super().__init__(master, bg="#2C3E50", padx=10, pady=5)
        self.app = app
        self.create_buttons()

    def create_buttons(self):
        buttons = ["Home", "Upload", "Data Cleanup"," Visualization", "Classification", "Export"]
        for btn_text in buttons:
            btn = ctk.CTkButton(self, text=btn_text, fg_color="#3498DB", command=lambda text=btn_text: self.button_click(text))
            btn.pack(side=tk.TOP, fill=tk.X, padx=60, pady=50)

    def button_click(self, text):
        self.app.on_button_click(text)

if __name__ == "__main__":
    app = CustomApp(root_tk)
    app.create_gui()
    root_tk.mainloop()
