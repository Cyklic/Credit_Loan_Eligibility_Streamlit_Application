# from setuptools import find_packages, setup


# setup(
#     name='src',
#     packages=find_packages(),
#     version='0.1.0',
#     description='Credit Risk Model code structuring',
#     author='Leonard Umoru',
#     license='',
# )

import logging
import traceback
from src.data.make_dataset import load_and_preprocess_data
from src.visualization.visualize import plot_correlation_heatmap, plot_feature_importance
from src.features.build_features import create_dummy_vars
from src.models.train_model import train_RFmodel
from src.models.predict_model import evaluate_model

# Configure logging
logging.basicConfig(
    filename="credit_pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="a"
)

if __name__ == "__main__":
    try:
        logging.info("Credit Loan Classification Pipeline started.")

        # Load and preprocess the data
        data_path = "data/raw/credit.csv"
        try:
            df = load_and_preprocess_data(data_path)
            logging.info(f"Data loaded and preprocessed from {data_path}")
        except Exception as e:
            logging.error("Error loading and preprocessing the dataset.")
            logging.error(traceback.format_exc())
            raise

        # Create dummy variables and separate features and target
        try:
            X, y = create_dummy_vars(df)
            logging.info("Dummy variables created successfully.")
        except Exception as e:
            logging.error("Error during feature engineering.")
            logging.error(traceback.format_exc())
            raise

        # Train the Random Forest model
        try:
            model, X_test_scaled, y_test = train_RFmodel(X, y)
            logging.info("Random Forest model trained successfully.")
        except Exception as e:
            logging.error("Error during model training.")
            logging.error(traceback.format_exc())
            raise

        # Plot feature importance
        try:
            plot_feature_importance(model, X)
            logging.info("Feature importance plot generated.")
        except Exception as e:
            logging.warning("Failed to generate feature importance plot.")
            logging.warning(traceback.format_exc())

        # Evaluate the model
        try:
            accuracy, confusion_mat = evaluate_model(model, X_test_scaled, y_test)
            logging.info(f"Model evaluated successfully. Accuracy: {accuracy}")
            logging.info(f"Confusion Matrix:\n{confusion_mat}")
            print(f"Accuracy: {accuracy}")
            print(f"Confusion Matrix:\n{confusion_mat}")
        except Exception as e:
            logging.error("Error during model evaluation.")
            logging.error(traceback.format_exc())
            raise

        logging.info("Credit Loan Classification Pipeline completed successfully.")

    except Exception as e:
        logging.critical("Pipeline execution failed.")
        logging.critical(traceback.format_exc())

