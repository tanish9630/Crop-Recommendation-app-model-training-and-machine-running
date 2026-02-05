# Crop Recommendation System

This project implements a Machine Learning-based system to recommend the most suitable crops for cultivation based on soil and environmental parameters. It includes both a data analysis notebook and a Flask-based web application.

## 📌 Project Overview

Agriculture is the backbone of many economies. This project aims to assist farmers by recommending the best crop to grow on their land, thereby maximizing yield and minimizing losses. The system analyzes various parameters such as Nitrogen, Phosphorous, Potassium (NPK) levels, temperature, humidity, pH, and rainfall to predict the optimal crop.

## 🚀 Features

-   **Data Analysis & Modeling:** Comprehensive data analysis and model training using Jupyter Notebooks.
-   **Machine Learning Models:** Utilizes algorithms like Logistic Regression, Random Forest, and XGBoost.
-   **Web Application:** A user-friendly web interface built with Flask to input soil data and get crop recommendations.
-   **Accurate Predictions:** Recommends the best crop based on scientific data.

## 📂 Project Structure

-   `Crop recommendation.ipynb`: Jupyter notebook containing data exploration, visualization, and model training steps.
-   `Crop_recommendationV2.csv`: The dataset used for training and testing the models.
-   `Crop_Recommendation_App-main/`: Directory containing the Flask web application code.
    -   `app.py`: Main Flask application file.
    -   `model.pkl`: Trained machine learning model.
    -   `templates/`: HTML templates for the web app.
    -   `static/`: Static assets (CSS, images).
-   `Crop_Recommendation_Report.pdf`: Detailed project report.

## 🛠️ Installation & Usage

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/tanish9630/Crop-Recommendation-app-model-training-and-machine-running.git
    cd Crop-Recommendation-app-model-training-and-machine-running
    ```

2.  **Install Dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost flask
    ```
    *Note: You may need to install specific versions if mentioned in a requirements.txt file.*

3.  **Run the Web Application:**
    Navigate to the app directory if necessary and run:
    ```bash
    cd "Crop_Recommendation_App-main"
    python app.py
    ```
    The app should now be running on `http://127.0.0.1:5000/`.

4.  **Explore the Notebook:**
    Open `Crop recommendation.ipynb` using Jupyter Notebook or JupyterLab to see the data analysis and model building process.

## 📊 Dataset

The dataset used in this project is `Crop_recommendationV2.csv`. It contains the following features:

-   **N**: Ratio of Nitrogen content in soil
-   **P**: Ratio of Phosphorous content in soil
-   **K**: Ratio of Potassium content in soil
-   **temperature**: Temperature in degree Celsius
-   **humidity**: Relative humidity in %
-   **ph**: ph value of the soil
-   **rainfall**: Rainfall in mm
-   **label**: The recommended crop (Target variable)

## 🤝 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.
