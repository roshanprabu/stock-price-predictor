# Stock Price Prediction with LSTM

## Project Overview

This project implements a Long Short-Term Memory (LSTM) neural network model to predict future stock prices based on historical data. The goal is to demonstrate the application of deep learning techniques, specifically LSTMs, in time series forecasting within the financial domain. While LSTMs are powerful for sequential data, it's important to note that stock market prediction is inherently challenging due to its volatility and numerous external factors. This model serves as an analytical tool and an educational example, not financial advice.

## Features

* **Historical Data Acquisition:** Fetches real-time and historical stock data from Yahoo Finance for any specified ticker.

* **Data Preprocessing:**

  * Scales data (MinMaxScaler) to prepare it for neural network input.

  * Transforms time series data into sequences (sliding windows) suitable for LSTM.

  * Performs a chronological train-test split to ensure realistic evaluation.

* **LSTM Model:**

  * A deep learning model built with stacked LSTM layers and Dropout for regularization.

  * Trained using Mean Squared Error (MSE) loss and Adam optimizer.

* **Prediction & Evaluation:**

  * Generates future stock price predictions.

  * Evaluates model performance using key metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

* **Visualization:**

  * Plots actual vs. predicted stock prices for visual comparison.

  * Displays training and validation loss/error curves to monitor learning progress and identify overfitting.

## Technologies Used

* **Python:** The core programming language.

* **`yfinance`:** For downloading historical stock data.

* **`pandas`:** For data manipulation and analysis.

* **`numpy`:** For numerical operations.

* **`tensorflow` / `keras`:** For building and training the LSTM neural network.

* **`scikit-learn`:** For data preprocessing (e.g., `MinMaxScaler`).

* **`matplotlib` & `seaborn`:** For data visualization.

## How to Run This Project

To run this Jupyter Notebook locally or in Google Colab, follow these steps:

### 1. Setup Environment

**Option A: Google Colab (Recommended for quick start)**

* Go to <https://colab.research.google.com/>.

* Click `File` -> `New notebook`.

* Copy and paste the entire code from the `.ipynb` file into the first cell.

* Ensure the `!pip install` line at the top of the script is uncommented (`!pip install yfinance ...`).

**Option B: Local Jupyter Notebook**

* Ensure you have Python installed.

* Install Jupyter Notebook: `pip install notebook`

* Install all required libraries:

pip install yfinance pandas numpy tensorflow matplotlib seaborn scikit-learn


* Launch Jupyter: `jupyter notebook` from your terminal in the project directory.

* Open the `.ipynb` file you uploaded. The `!pip install` line should remain commented out if you installed libraries globally.

### 2. Execution

* **Run all cells sequentially.** In Colab, you can click `Runtime` -> `Run all`. In Jupyter, `Cell` -> `Run All`.

* The script will:

* Download historical data for the specified `STOCK_TICKER`.

* Preprocess the data.

* Train the LSTM model.

* Generate predictions.

* Display evaluation metrics and plots.

### 3. Configuration

You can easily modify the following variables in the script to experiment with different stocks or parameters:

STOCK_TICKER = 'AAPL'       # Change to 'GOOG', 'MSFT', etc.
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
SEQUENCE_LENGTH = 60        # Number of past days to consider for prediction
EPOCHS = 50                 # Number of training cycles
BATCH_SIZE = 32             # Samples per training batch


## Expected Results

Upon successful execution, the notebook will output:

* Confirmation messages for each processing phase.

* Data summaries (`df.head()`, `df.info()`).

* Model architecture summary (`model.summary()`).

* Training progress (loss and MAE for each epoch).

* Final evaluation metrics (MSE, RMSE, MAE) on the test data.

* Two plots:

  1. **Actual vs. Predicted Prices:** A visual comparison showing how closely the model's predictions align with the true stock prices on the test set.

  2. **Model Loss and MAE Over Epochs:** Graphs illustrating the learning progress and helping to identify potential overfitting.

## Challenges and Limitations

During the development of this project, I found the inherent unpredictability of the stock market to be the most challenging aspect. Even with advanced models like LSTMs, factors outside of historical price data (like breaking news or sudden economic shifts) significantly influence stock movements, making truly accurate long-term predictions extremely difficult. This project reinforced for me that while deep learning can identify complex patterns, it's a tool for analysis and understanding trends, not a crystal ball for guaranteed future outcomes.

Market Volatility: Stock markets are highly unpredictable due to numerous external factors (news, economic events, geopolitical shifts).

Efficient Market Hypothesis: This model captures historical patterns but does not guarantee future accuracy or profitability. It's a tool for analysis, not a crystal ball.

Data Scope: This model primarily uses historical price data. Incorporating external factors like news sentiment or economic indicators could enhance performance but adds complexity.

