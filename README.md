#Credit Risk Prediction Using Neural Networks
Overview

This project predicts the credit risk of loan applicants using a feedforward neural network. The model is trained on the Lending Club dataset, which contains detailed information about accepted and rejected loan applications. The goal is to classify whether a loan will be "Fully Paid" or result in "Default/Charged Off" based on applicant and loan features.
Dataset

    Source: Lending Club Loan Data (Kaggle)

    Files Used:

        accepted_2007_to_2018Q4.csv: Contains accepted loan applications with loan status.

        rejected_2007_to_2018Q4.csv: Contains rejected loan applications (not used for supervised learning).

Project Structure

text
.
├── accepted_2007_to_2018Q4.csv
├── rejected_2007_to_2018Q4.csv
├── credit_risk_prediction.ipynb
├── credit_risk_model.h5
└── README.md

How to Run

    Environment:
    This project is designed for Google Colab but can run on any environment with Python 3.x.

    Dependencies:
    The following Python libraries are required:

        pandas

        numpy

        matplotlib

        seaborn

        scikit-learn

        tensorflow

        keras

    In Colab, these are installed automatically by the notebook.

    Data Preparation:

        Download the Lending Club dataset from Kaggle.

        Upload accepted_2007_to_2018Q4.csv and rejected_2007_to_2018Q4.csv to your Colab or working directory.

    Run the Notebook:

        Open credit_risk_prediction.ipynb in Colab.

        Execute all cells.
        The notebook:

            Loads and preprocesses the data

            Trains a feedforward neural network

            Evaluates model performance

            Compares with logistic regression

            Saves the trained model as credit_risk_model.h5

Model Details

    Input Features: Loan amount, term, interest rate, grade, employment length, home ownership, annual income, purpose, DTI, delinquencies, revolving utilization, total accounts, etc.

    Target: loan_status (1 = Default/Charged Off, 0 = Fully Paid)

    Architecture:

        Dense (64 units, ReLU) + Dropout

        Dense (32 units, ReLU) + Dropout

        Output: Dense (1 unit, Sigmoid)

    Loss: Binary Crossentropy

    Optimizer: Adam

Results

    The model outputs accuracy, classification report, and confusion matrix.

    Performance is compared with a baseline logistic regression model.

Notes

    The accepted loans file is very large. For initial experiments, the code samples 100,000 rows. Remove the nrows parameter to use the full dataset (requires more RAM).

    The rejected loans file is not used in supervised training as it lacks the target variable.

    For further analysis, you can explore unsupervised or semi-supervised learning with rejected loans.

License

This project is for educational and research purposes. Please check the Lending Club dataset license for data usage terms.
Acknowledgements

    Lending Club and Kaggle for the dataset.

    TensorFlow/Keras and scikit-learn for machine learning libraries.

    The open-source data science community.
