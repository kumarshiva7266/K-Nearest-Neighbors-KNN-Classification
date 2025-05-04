# KNN Classification GUI

This is an advanced GUI application for K-Nearest Neighbors (KNN) classification that allows you to experiment with different datasets, features, and parameters.

## Features

- Multiple built-in datasets (Iris, Wine, Breast Cancer)
- Interactive feature selection
- Adjustable K value (1-20)
- Real-time visualization of decision boundaries
- Confusion matrix visualization
- Detailed classification metrics
- Data normalization
- Interactive plots

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python knn_gui.py
```

2. Using the GUI:
   - Select a dataset from the dropdown menu
   - Click "Load Dataset" to load the selected dataset
   - Choose which features to use by checking/unchecking the feature checkboxes
   - Adjust the K value using the slider
   - Click "Train Model" to train the KNN classifier
   - Use "Show Metrics" to view the confusion matrix and classification report
   - Use "Plot Decision Boundary" to visualize the decision boundaries (select exactly 2 features first)

## Notes

- For decision boundary visualization, you must select exactly 2 features
- The application automatically normalizes the data using StandardScaler
- The train/test split is set to 80/20 with a random state of 42
- All visualizations are interactive and can be zoomed/panned 