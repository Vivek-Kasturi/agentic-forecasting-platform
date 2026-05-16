from data_pipeline.load_dataset import load_walmart_dataset
from data_pipeline.validation import run_validation
from utils.logger import logger


def main():
    logger.info("🚀 Agentic Forecasting Platform initialized")

    # Step 1: Load data
    df = load_walmart_dataset()

    logger.info(f"Dataset loaded with shape: {df.shape}")

    # Step 2: Validate data
    run_validation(df)

    # Step 3: Placeholder for next stages (feature engineering / model)
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()