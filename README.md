# Collaborative Sentiment Analysis Pipeline using BERT

## Project Overview

This project, developed as part of an **MLOps project**, implements a complete Sentiment Analysis Pipeline using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. The pipeline processes app reviews from the Google Play Store, classifies sentiments into three categories—negative (0), neutral (1), and positive (2)—and provides an interactive web interface for real-time predictions using Streamlit. The project simulates a professional MLOps workflow, emphasizing best practices such as version control with Git, collaborative development through pull requests and reviews, unit testing with `pytest`, automation scripts using `setup.sh` and `Makefile`, and comprehensive documentation.

The dataset used in this project is sourced from Kaggle: **Google Play Store Reviews**, a collection of app reviews for sentiment analysis. You can explore the dataset and related work here: [Sentiment Analysis using BERT on Kaggle](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert). If you’re interested in diving deeper into the project or the dataset, the Kaggle link provides additional insights and resources.


## About the Dataset

### Google Play Store Reviews
- **Source**: [Kaggle - Google Play Store Reviews](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert)
- **Description**: This dataset contains app reviews collected from the Google Play Store, intended for the task of sentiment analysis. The Google Play Store, formerly known as Android Market, is a digital distribution service operated and developed by Google. It serves as the official app store for certified devices running on the Android operating system, allowing users to browse and download applications developed with the Android software development kit (SDK) and published through Google. With over 82 billion app downloads and more than 3.5 million published apps, it is the largest app store in the world.
- **Last Updated**: 4 years ago (Version 1)
- **Usage in Project**: The dataset is loaded from `/Users/johannafokui/Downloads/dataset.csv` on Johanna’s machine. For testing on other machines, you can download the dataset from the Kaggle link above and place it in the appropriate directory, or create a temporary `dataset.csv` file with the required columns (`content` and `score`).


## 🔹 Project Structure

The project is organized to reflect a professional MLOps workflow, with clear separation of concerns between source code, tests, and automation scripts. Below is the structure of the repository:

```
Sentiment-Analysis-BERT/
├── models/                    # Directory for saved BERT model files
├── src/                       # Source code for the pipeline
│   ├── data_extraction.py     # Loads and preprocesses the dataset
│   ├── data_processing.py     # Handles text cleaning, tokenization, and data splitting
│   ├── model.py               # Trains the BERT model for sentiment analysis
│   └── inference.py           # Performs inference using the trained model
├── tests/                     # Unit tests for the pipeline
│   └── unit/
│       ├── test_data_extraction.py  # Tests for data extraction
│       ├── test_data_processing.py  # Tests for data processing
│       ├── test_model.py            # Tests for model training
│       └── test_inference.py        # Tests for inference
├── app.py                     # Streamlit web app for real-time predictions
├── setup.sh                   # Script to automate environment setup
├── Makefile                   # Makefile with commands for setup, testing, running, and launching the app
├── README.md                  # Project documentation (this file)
├── report.md                  # Detailed project report
├── requirements.txt           # List of Python dependencies
└── .gitignore                 # Git ignore file for excluding unnecessary files
```


## ⚙️ Setup

Follow these steps to set up the project on your local machine:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/JoNoubs/Sentiment-Analysis-BERT.git
   cd Sentiment-Analysis-BERT
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv sentiment_env
   source sentiment_env/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   - **Note**: If you encounter a `tf-keras` dependency error (e.g., `ModuleNotFoundError: No module named 'tf_keras'`), install it manually:
     ```bash
     pip install tf-keras
     ```

4. **Verify Installation**:
   - Ensure all dependencies are installed correctly:
     ```bash
     python3 -c "import transformers, torch, pandas, numpy, sklearn, pytest, streamlit, pytest_mock; print('All imports successful')"
     ```
     - **Expected Output**: `All imports successful`

5. **Dataset Setup**:
   - The pipeline expects the dataset at `/Users/johannafokui/Downloads/dataset.csv`. If you’re running on a different machine:
     - Download the dataset from [Kaggle - Google Play Store Reviews](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert) and place it in the appropriate directory.
     - Alternatively, create a temporary `dataset.csv` file for testing:
       ```bash
       echo "content,score\nGood movie,5\nBad film,1\nAmazing story,4\nBad plot,2\nOkay experience,3" > dataset.csv
       ```
     - Update the path in `src/data_extraction.py` to point to your dataset:
       ```bash
       nano src/data_extraction.py
       ```
       - Change the `path` parameter in the `load_data()` function to your dataset’s location, e.g.:
         ```python
         def load_data(path="dataset.csv"):
         ```
       - Save and exit.


## Usage

The project provides several ways to interact with the sentiment analysis pipeline:

- **Train the Model**:
  - Train the BERT model on the dataset:
    ```bash
    python src/model.py
    ```
  - This will load the dataset, preprocess and tokenize the data, train the model for 2 epochs, and save the trained model to the `models/` directory.

- **Perform Inference**:
  - Use the trained model to predict the sentiment of a sample text:
    ```bash
    python src/inference.py
    ```
  - **Expected Output**: `Sentiment for 'This movie is amazing!': Positive`

- **Launch the Web App**:
  - Start the Streamlit web app for real-time sentiment predictions:
    ```bash
    streamlit run app.py
    ```
  - Open your browser and go to `http://localhost:8501` to interact with the app. Enter a text (e.g., “This movie is great!”) and click “Predict” to see the sentiment.


## Makefile Commands

The project includes a `Makefile` to simplify common tasks. Use the following commands:

- `make setup`: Sets up the virtual environment and installs dependencies.
  ```bash
  make setup
  ```

- `make test`: Runs all unit tests to verify the pipeline components.
  ```bash
  make test
  ```

- `make run`: Trains the model and performs inference on a sample text.
  ```bash
  make run
  ```

- `make app`: Launches the Streamlit web app.
  ```bash
  make app
  ```


## MLOps Workflow

This project was developed as part of an MLOps project, simulating a professional software development environment. Key MLOps practices implemented include:

- **Version Control**: Used Git for version control, with feature branches (`feature-data-extraction`, `feature-data-cleaning`, etc.) and pull requests for collaborative development.
- **Collaboration**: Johanna and Armel collaborated through GitHub pull requests and reviews, ensuring code quality and shared understanding.
- **Unit Testing**: Comprehensive unit tests were written using `pytest` to validate each component of the pipeline (`data_extraction`, `data_processing`, `model`, `inference`).
- **Automation**: Automated setup and execution with `setup.sh` and `Makefile`, streamlining the development and deployment process.
- **Documentation**: Detailed documentation in `README.md` and `report.md`, covering the project overview, setup, usage, and contributions.
- **Deployment**: Deployed a user-friendly Streamlit web app for real-time sentiment predictions.


## Explore More

If you’d like to explore the project further, check out the Kaggle notebook where the dataset was sourced and additional sentiment analysis work was done: [Sentiment Analysis using BERT on Kaggle](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert). The notebook provides insights into the dataset and alternative approaches to sentiment analysis using BERT.


## Contributors

- **Student A: Johanna (JoNoubs)**  
  - Responsibilities: Data extraction, text cleaning, web app interface, documentation lead.
  - GitHub: [JoNoubs](https://github.com/JoNoubs)

- **Student B: Armel (m-armel)**  
  - Responsibilities: Tokenization, data splitting, model training, inference, automation scripts.
  - GitHub: [m-armel](https://github.com/m-armel)


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (if applicable).


## Acknowledgments

- Thanks to Kaggle for providing the Google Play Store Reviews dataset.
- Special thanks to the Hugging Face team for the `transformers` library, which made working with BERT seamless.
- Gratitude to the Streamlit team for their excellent framework for building interactive web apps.

