import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import customtkinter as ctk
from PIL import Image, ImageTk
import io

class KNNClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced KNN Classification GUI")
        self.root.geometry("1400x900")
        
        # Set theme and color scheme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize variables
        self.dataset = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.current_plot = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main container
        self.main_container = ctk.CTkFrame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create left panel for controls
        self.control_frame = ctk.CTkFrame(self.main_container, width=300)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Create right panel for visualization
        self.visualization_frame = ctk.CTkFrame(self.main_container)
        self.visualization_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ctk.CTkLabel(self.control_frame, text="KNN Classifier", font=("Helvetica", 24, "bold"))
        title_label.pack(pady=20)
        
        # Dataset selection
        dataset_frame = ctk.CTkFrame(self.control_frame)
        dataset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(dataset_frame, text="Select Dataset:", font=("Helvetica", 14)).pack(pady=5)
        self.dataset_var = ctk.StringVar(value="iris")
        datasets = ["iris", "wine", "breast_cancer"]
        self.dataset_combo = ctk.CTkComboBox(dataset_frame, values=datasets, variable=self.dataset_var)
        self.dataset_combo.pack(pady=5)
        
        # K value selection
        k_frame = ctk.CTkFrame(self.control_frame)
        k_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(k_frame, text="K Value:", font=("Helvetica", 14)).pack(pady=5)
        self.k_var = ctk.IntVar(value=5)
        self.k_scale = ctk.CTkSlider(k_frame, from_=1, to=20, number_of_steps=19, variable=self.k_var)
        self.k_scale.pack(pady=5)
        self.k_label = ctk.CTkLabel(k_frame, text="5")
        self.k_label.pack()
        self.k_scale.configure(command=self.update_k_label)
        
        # Feature selection
        feature_frame = ctk.CTkFrame(self.control_frame)
        feature_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(feature_frame, text="Select Features:", font=("Helvetica", 14)).pack(pady=5)
        self.feature_frame = ctk.CTkScrollableFrame(feature_frame, height=200)
        self.feature_frame.pack(fill=tk.X, pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(self.control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ctk.CTkButton(button_frame, text="Load Dataset", command=self.load_dataset).pack(pady=5, fill=tk.X)
        ctk.CTkButton(button_frame, text="Train Model", command=self.train_model).pack(pady=5, fill=tk.X)
        ctk.CTkButton(button_frame, text="Show Metrics", command=self.show_metrics).pack(pady=5, fill=tk.X)
        ctk.CTkButton(button_frame, text="Plot Decision Boundary", command=self.plot_decision_boundary).pack(pady=5, fill=tk.X)
        ctk.CTkButton(button_frame, text="Cross Validation", command=self.show_cross_validation).pack(pady=5, fill=tk.X)
        
        # Results text
        results_frame = ctk.CTkFrame(self.control_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(results_frame, text="Results:", font=("Helvetica", 14)).pack(pady=5)
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
    def update_k_label(self, value):
        self.k_label.configure(text=str(int(float(value))))
        
    def load_dataset(self):
        dataset_name = self.dataset_var.get()
        if dataset_name == "iris":
            self.dataset = load_iris()
        elif dataset_name == "wine":
            self.dataset = load_wine()
        else:
            self.dataset = load_breast_cancer()
            
        self.X = self.dataset.data
        self.y = self.dataset.target
        
        # Update feature selection checkboxes
        for widget in self.feature_frame.winfo_children():
            widget.destroy()
            
        self.feature_vars = []
        for i, feature in enumerate(self.dataset.feature_names):
            var = ctk.BooleanVar(value=True)
            self.feature_vars.append(var)
            ctk.CTkCheckBox(self.feature_frame, text=feature, variable=var).pack(anchor=tk.W, pady=2)
            
        messagebox.showinfo("Success", f"Loaded {dataset_name} dataset successfully!")
        
    def train_model(self):
        if self.X is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return
            
        # Get selected features
        selected_features = [i for i, var in enumerate(self.feature_vars) if var.get()]
        if len(selected_features) < 2:
            messagebox.showerror("Error", "Please select at least 2 features!")
            return
            
        X_selected = self.X[:, selected_features]
        
        # Split and scale data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_selected, self.y, test_size=0.2, random_state=42
        )
        
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        self.model = KNeighborsClassifier(n_neighbors=self.k_var.get())
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Model trained successfully!\nAccuracy: {accuracy:.4f}\n")
        
    def show_metrics(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Create confusion matrix plot
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        self.update_plot()
        
        # Show classification report
        report = classification_report(self.y_test, y_pred)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Classification Report:\n{report}")
        
    def plot_decision_boundary(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        # Get selected features
        selected_features = [i for i, var in enumerate(self.feature_vars) if var.get()]
        if len(selected_features) != 2:
            messagebox.showerror("Error", "Please select exactly 2 features for visualization!")
            return
            
        X_selected = self.X[:, selected_features]
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Create mesh grid
        x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
        y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        # Predict for mesh grid
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=self.y, alpha=0.8)
        plt.xlabel(self.dataset.feature_names[selected_features[0]])
        plt.ylabel(self.dataset.feature_names[selected_features[1]])
        plt.title('Decision Boundary')
        
        self.update_plot()
        
    def show_cross_validation(self):
        if self.model is None:
            messagebox.showerror("Error", "Please train the model first!")
            return
            
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, self.X_train_scaled, self.y_train, cv=5)
        
        # Plot cross-validation scores
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, 6), cv_scores)
        plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy')
        plt.title('Cross-Validation Scores')
        plt.legend()
        
        self.update_plot()
        
        # Show results
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, f"Cross-Validation Results:\n")
        self.results_text.insert(tk.END, f"Mean CV Score: {cv_scores.mean():.4f}\n")
        self.results_text.insert(tk.END, f"Std CV Score: {cv_scores.std():.4f}\n")
        
    def update_plot(self):
        # Clear previous plot
        for widget in self.visualization_frame.winfo_children():
            widget.destroy()
            
        # Create canvas and toolbar
        canvas = FigureCanvasTkAgg(plt.gcf(), master=self.visualization_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.visualization_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = ctk.CTk()
    app = KNNClassifierGUI(root)
    root.mainloop() 