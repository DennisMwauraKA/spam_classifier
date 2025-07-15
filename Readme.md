# SMS Spam Classifier

This project is a machine learning-based classifier that detects whether an SMS message is spam or not (ham). It uses Python, scikit-learn, and text processing techniques like TF-IDF to train and evaluate a spam detection model.

---

## Project Overview

The classifier performs the following tasks:

1. Load and clean SMS data from a CSV file
2. Preprocess and vectorize the text data
3. Train a Multinomial Naive Bayes classifier
4. Evaluate the model using accuracy, precision, recall, and F1 score
5. Visualize the class distribution using Seaborn

---

## Dataset

The dataset used is `spam.csv`, which contains:

- Column `v1`: the label (`ham` or `spam`)
- Column `v2`: the message content

---

## Technologies Used

- Python 3.x
- scikit-learn
- pandas
- matplotlib
- seaborn

---

## Installation and Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/spam-classifier.git
    cd spam_classifier
    ```

2. Create and activate a virtual environment:

    ```bash
    uv venv .venv
    ```

    **On Windows:**

    ```bash
    .venv\Scripts\activate
    ```

    **On macOS/Linux:**

    ```bash
    source .venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    uv pip install -r requirements.txt
    ```

4. Run the project:

    ```bash
    python main.py
    ```

---

## Output

The script prints evaluation metrics and shows a bar chart:

- Accuracy
- Precision
- Recall
- F1 Score
- Class distribution chart (Ham vs Spam)

---



## License

This project is licensed under the MIT License.

---

## Author

Dennis Kariuki  
AI Engineering Student  
Kenya
