
# PRICE PERFORMANCE PREDICTION

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaoji (Jay) Wang - SPRING 2024 Semester  
**Author:** [Shivaramakrishna Reddy Kasireddy](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/README.md)


## Background

### 1. What is it about?

The report is centered on "Price Performance Prediction," specifically analyzing Microsoft (MSFT) and JP Morgan (JPM) stocks within the S&P 500 framework. It delves into the relationship between these stocks and the S&P 500 index, exploring their performance using historical data over the past decade. By leveraging financial modeling techniques, the analysis seeks to uncover correlations, calculate beta coefficients, and compare the Sharpe Ratios of these two stocks against the S&P 500.

### 2. Why does it matter?

This analysis matters because accurate stock price prediction is crucial in finance, enabling investors, traders, and financial institutions to make better-informed decisions. Understanding potential future price movements can lead to effective risk management, optimized investment strategies, and minimized losses. The correlations and comparative insights between Microsoft, JP Morgan, and the S&P 500 can help stakeholders maximize their returns while navigating the financial markets strategically.

### 3. What are your research questions?

1. How strongly are Microsoft and JP Morgan correlated with the S&P 500 index, and what does that imply about their performance?
2. How do Microsoft and JP Morgan compare in terms of beta coefficients, and what does this indicate regarding their volatility and market sensitivity?
3. What are the Sharpe Ratios of these two stocks compared to the S&P 500, and what does this suggest about their risk-adjusted returns?
4. How well do different machine learning models predict and compare the performance of these stocks?

### 4. Why is it relevant?

This analysis is relevant because it provides practical insights for stakeholders looking to maximize their returns while minimizing risk in today's dynamic market. The comparative study helps investors choose the appropriate stocks based on their risk appetite and investment strategy. Financial models, including ARIMA, LSTM, and others, offer a multi-dimensional perspective for understanding the drivers of market movements, contributing to the broader field of financial data science.

## Dataset:

**Dataset source:** Yahoo Finance (S&P 500 (^GSPC)), Microsoft (MSFT), and JP Morgan (JPM) historical data for the past 10 years  
**Link:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EGSPC/history?period1=1550188800&period2=1707955200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)  
**Size:** 25 - 65 MB  
**Rows:** 2516 (all files together)  
**Columns:** 6

## Exploratory Data Analysis (EDA)

### Overview

Exploratory Data Analysis (EDA) is a crucial step in understanding the dataset's structure, distribution, and relationships between variables. Let's explore the real estate dataset to gain insights into its features and potential patterns.

### Data Visualization

### Data Cleansing:

**Handling Missing Values:**  
- No missing values were detected in the dataset, ensuring that all records contain complete information.

**Handling Duplicate Rows:**  
- No duplicate rows were found in the dataset, indicating data consistency and preventing redundancy.

---

## Price Performance Prediction: Analyzing Microsoft and JP Morgan in the S&P 500 Framework

### Introduction:

This project aims to analyze the price performance of Microsoft (MSFT) and JP Morgan (JPM) stocks within the S&P 500 framework, drawing insights from historical market data spanning the last decade. The study leverages machine learning models and statistical techniques to uncover correlations, beta coefficients, and Sharpe Ratios, providing a comprehensive understanding of their relationship with the S&P 500 index.

### Significance of the Study:

Accurate prediction of stock prices is critical for investors, traders, and financial institutions as it helps optimize strategies and manage risks. By forecasting potential future price movements, stakeholders can make better-informed decisions regarding buying, selling, or holding assets.

