<<<<<<< HEAD


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
├── setup.sh                   # Script to automate environment setup (Unix-based systems)
├── Makefile                   # Makefile with commands for setup, testing, running, and launching the app (Unix-based systems)
├── README.md                  # Project documentation (this file)
├── report.md                  # Detailed project report
├── requirements.txt           # List of Python dependencies
└── .gitignore                 # Git ignore file for excluding unnecessary files
```



## ⚙️ Setup

Follow these steps to set up the project on your local machine. Instructions are provided for both Unix-based systems (Linux, macOS) and Windows.

### For Unix-based Systems (Linux, macOS)

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
   - **Note for macOS Users**: If you encounter a `permission denied` error with `source`, use:
     ```bash
     . sentiment_env/bin/activate
     ```

3. **Install Dependencies**:
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - **Important**: Using the `Trainer` with PyTorch (as in `src/model.py`) requires `accelerate>=0.26.0`. If you encounter an error like:
     ```
     ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`
     ```
     Install the `accelerate` package:
     ```bash
     pip install 'accelerate>=0.26.0'
     ```
   - If you encounter a `tf-keras` dependency error (e.g., `ModuleNotFoundError: No module named 'tf_keras'`), install it manually:
     ```bash
     pip install tf-keras
     ```

4. **Verify Installation**:
   - Ensure all dependencies are installed correctly:
     ```bash
     python3 -c "import transformers, torch, pandas, numpy, sklearn, pytest, streamlit, pytest_mock, accelerate; print('All imports successful')"
     ```
     - **Expected Output**: `All imports successful`

### For Windows

1. **Clone the Repository**:
   - Open a command prompt (cmd) or PowerShell:
     ```cmd
     git clone https://github.com/JoNoubs/Sentiment-Analysis-BERT.git
     cd Sentiment-Analysis-BERT
     ```

2. **Create and Activate a Virtual Environment**:
   - Create the virtual environment:
     ```cmd
     python -m venv sentiment_env
     ```
   - Activate the virtual environment:
     - In Command Prompt:
       ```cmd
       sentiment_env\Scripts\activate
       ```
     - In PowerShell:
       ```powershell
       .\sentiment_env\Scripts\Activate.ps1
       ```
   - **Verification**:
     - Your prompt should change to include `(sentiment_env)`.

3. **Install Dependencies**:
   - Install the required packages:
     ```cmd
     pip install -r requirements.txt
     ```
   - **Important**: Using the `Trainer` with PyTorch requires `accelerate>=0.26.0`. If you encounter an error like:
     ```
     ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]` or `pip install 'accelerate>=0.26.0'`
     ```
     Install the `accelerate` package:
     ```cmd
     pip install "accelerate>=0.26.0"
     ```
   - If you encounter a `tf-keras` dependency error, install it:
     ```cmd
     pip install tf-keras
     ```

4. **Verify Installation**:
   - Ensure all dependencies are installed correctly:
     ```cmd
     python -c "import transformers, torch, pandas, numpy, sklearn, pytest, streamlit, pytest_mock, accelerate; print('All imports successful')"
     ```
     - **Expected Output**: `All imports successful`

5. **Dataset Setup**:
   - The pipeline expects the dataset at `/Users/johannafokui/Downloads/dataset.csv` (Johanna’s machine). For testing on your machine:
     - Download the dataset from [Kaggle - Google Play Store Reviews](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert) and place it in a directory of your choice (e.g., `C:\Users\YourUsername\Downloads\dataset.csv` on Windows).
     - Alternatively, create a temporary `dataset.csv` file for testing:
       - In Command Prompt:
         ```cmd
         echo content,score> dataset.csv
         echo Good movie,5>> dataset.csv
         echo Bad film,1>> dataset.csv
         echo Amazing story,4>> dataset.csv
         echo Bad plot,2>> dataset.csv
         echo Okay experience,3>> dataset.csv
         ```
       - In PowerShell:
         ```powershell
         "content,score`nGood movie,5`nBad film,1`nAmazing story,4`nBad plot,2`nOkay experience,3" | Out-File -FilePath dataset.csv -Encoding utf8
         ```
     - Update the path in `src/data_extraction.py` to point to your dataset’s location:
       - Open `src/data_extraction.py` in a text editor (e.g., Notepad on Windows):
         ```cmd
         notepad src\data_extraction.py
         ```
       - Change the `path` parameter in the `load_data()` function to your dataset’s location, e.g.:
         ```python
         def load_data(path="dataset.csv"):
         ```
       - Save and exit.



## Usage

The project provides several ways to interact with the sentiment analysis pipeline. The commands below are compatible with both Unix-based systems and Windows.

- **Train the Model**:
  - Train the BERT model on the dataset:
    ```bash
    python -m src.model
    ```
  - This will load the dataset, preprocess and tokenize the data, train the model for 2 epochs, and save the trained model to the `models/` directory.
  - **Note for Windows Users**: Use the same command in Command Prompt or PowerShell:
    ```cmd
    python -m src.model
    ```

- **Perform Inference**:
  - Use the trained model to predict the sentiment of a sample text:
    ```bash
    python -m src.inference
    ```
  - **Expected Output**: `Sentiment for 'This movie is amazing!': Positive`
  - **Note for Windows Users**: Use the same command:
    ```cmd
    python -m src.inference
    ```

- **Launch the Web App**:
  - Start the Streamlit web app for real-time sentiment predictions:
    ```bash
    streamlit run app.py
    ```
  - Open your browser and go to `http://localhost:8501` to interact with the app. Enter a text (e.g., “This movie is great!”) and click “Predict” to see the sentiment.
  - **Note for Windows Users**: Use the same command:
    ```cmd
    streamlit run app.py
    ```



## Makefile Commands (Unix-based Systems)

For Unix-based systems (Linux, macOS), the project includes a `Makefile` to simplify common tasks. Use the following commands:

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

### For Windows Users
- The `Makefile` is designed for Unix-based systems and may not work directly on Windows unless you have a Unix-like environment (e.g., WSL, Git Bash, or Cygwin). Instead, you can run the equivalent commands manually:
  - **Setup**:
    ```cmd
    python -m venv sentiment_env
    sentiment_env\Scripts\activate
    pip install -r requirements.txt
    pip install "accelerate>=0.26.0"
    pip install tf-keras
    ```
  - **Test**:
    ```cmd
    pytest tests\unit\ -v
    ```
  - **Run**:
    ```cmd
    python -m src.model
    python -m src.inference
    ```
  - **App**:
    ```cmd
    streamlit run app.py
    ```



## MLOps Workflow

This project was developed as part of an MLOps project, simulating a professional software development environment. Key MLOps practices implemented include:

- **Version Control**: Used Git for version control, with feature branches (`feature-data-extraction`, `feature-data-cleaning`, etc.) and pull requests for collaborative development.
- **Collaboration**: Johanna and Armel collaborated through GitHub pull requests and reviews, ensuring code quality and shared understanding.
- **Unit Testing**: Comprehensive unit tests were written using `pytest` to validate each component of the pipeline (`data_extraction`, `data_processing`, `model`, `inference`).
- **Automation**: Automated setup and execution with `setup.sh` and `Makefile` (for Unix-based systems), streamlining the development and deployment process.
- **Error Handling**: Improved error handling in `src/model.py` and `src/inference.py` to manage small datasets and runtime errors gracefully.
- **Documentation**: Detailed documentation in `README.md` and `report.md`, covering the project overview, setup, usage, and contributions.
- **Deployment**: Deployed a user-friendly Streamlit web app for real-time sentiment predictions.



## Explore More

If you’d like to explore the project further, check out the Kaggle notebook where the dataset was sourced and additional sentiment analysis work was done: [Sentiment Analysis using BERT on Kaggle](https://www.kaggle.com/code/prakharrathi25/sentiment-analysis-using-bert). The notebook provides insights into the dataset and alternative approaches to sentiment analysis using BERT.



## Acknowledgments

- Thanks to Kaggle for providing the Google Play Store Reviews dataset.
- Special thanks to the Hugging Face team for the `transformers` library and `accelerate` package, which made working with BERT seamless.
- Gratitude to the Streamlit team for their excellent framework for building interactive web apps.