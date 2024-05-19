
# PRICE PERFORMANCE PREDICTION

Prepared for UMBC Data Science Master Degree Capstone by Dr. Chaoji (Jay) Wang - SPRING 2024 Semester  
**Author:** [Shivaramakrishna Reddy Kasireddy](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/README.md)

[Linkedin](https://www.linkedin.com/in/shivakasireddy/)
[Presentation](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DATA/MG67053_CAPSTONE_PPT.pptx)
**Youtube Video**



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
![alt text](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/CORELATION.png)

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

# Streamlit:

- Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science.
- It helps in the project to create a web page and communicate with model for prediction.


![alt text](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/streamlit.png)
![alt text](https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/Streamlit2.png)

<img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/streamlit.png" alt="Streamlit Image 1" width="500" />
<img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/Streamlit2.png" alt="Streamlit Image 2" width="500" />

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/streamlit.png" alt="Streamlit Image 1" width="500" />
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/Streamlit2.png" alt="Streamlit Image 2" width="500" />
</div>

## Models Used for Prediction:

### 1. Linear Regression: 
Evaluates the relationship between stock price and independent variables.

- Useful for understanding the relationship between the dependent variable (like a stock price) and one or more independent variables. It’s simple and provides clear coefficients indicating the impact of each variable.

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/LR.png" alt="Streamlit Image 1" width="800" />
  
</div>

# 1.a Actual vs. Predicted Microsoft Prices

## Description:
- **X-axis:** S&P 500 Close Price
- **Y-axis:** Microsoft Close Price
- **Data Points:** Blue dots represent actual Microsoft closing prices.
- **Regression Line:** The red line represents predicted Microsoft prices based on the S&P 500 close prices.

## Insights:
1. **Positive Correlation:** There is a strong positive correlation between the S&P 500 close price and Microsoft's close price. This indicates that as the S&P 500 index increases, the price of Microsoft shares tends to increase as well.
2. **Linear Relationship:** The red line signifies a linear regression model, indicating a linear relationship between the S&P 500 and Microsoft prices.
3. **Model Fit:** The predicted line (red) aligns closely with the actual data points (blue), suggesting a good model fit. However, some variability around the line is observed, indicating that while the model is generally accurate, it doesn't capture all fluctuations.

# 1.b Linear Regression of JPM on S&P 500

## Description:
- **X-axis:** S&P 500 Close Price
- **Y-axis:** JP Morgan (JPM) Close Price
- **Data Points:** Blue dots represent actual JP Morgan closing prices.
- **Regression Line:** The red line represents the fitted values from the linear regression model.

## Insights:
1. **Positive Correlation:** Similar to Microsoft, there is a positive correlation between the S&P 500 close price and JP Morgan’s close price. This suggests that JP Morgan’s stock price increases with the S&P 500 index.
2. **Linear Relationship:** The red fitted line indicates a linear relationship, modeled using linear regression, between the S&P 500 and JP Morgan prices.
3. **Model Fit:** The fitted line adequately follows the overall trend of the data points, suggesting a reasonable model fit. However, there is notable scatter, indicating some level of variability not captured by the model.

---

- Both Microsoft and JP Morgan stock prices exhibit a positive linear correlation with the S&P 500 close prices, reflecting their movement in tandem with the broader market index.
- The linear regression models used for both stocks demonstrate good predictive capability, though some variability suggests the presence of additional influencing factors not accounted for in the models.
- These analyses underscore the utility of linear regression in predicting stock prices based on broader market indices, providing valuable insights for investors and analysts.




### 2. Moving Averages: 
Smoothens the data to identify trends.

- A moving average is a statistical technique used to analyze time series data by calculating averages of different subsets of the full data set. It is commonly used in financial markets to smooth out short-term fluctuations and highlight longer-term trends or cycles.

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/Movingaverages.png" alt="Streamlit Image 1" width="700" />
  
</div>

## S&P 500 Trends:
- Consistent upward trend from 2014 to 2024.
- Significant dip in early 2020 due to COVID-19, followed by a strong recovery.
- 50-day and 200-day moving averages indicate both short-term and long-term growth.

## Microsoft Trends:
- Strong upward trajectory, especially from 2016 onward.
- Similar dip in early 2020 with a quick recovery.
- 50-day and 200-day moving averages align with the overall growth trend.

## Moving Averages:
- 50-day averages reflect short-term trends, while 200-day averages show long-term trends.
- Both indices show moving averages aligning with actual prices, confirming growth.

## Investment Insights:
- Positive long-term growth for both S&P 500 and Microsoft.
- Moving averages are effective for identifying trends and making investment decisions.




### 3. ARIMA (Autoregressive Integrated Moving Average): 
Suitable for non-stationary data, forecasting based on past values and errors.

- Good for forecasting time series data based on its own past values (autoregressive) and a moving average of past forecast errors. It’s particularly useful for non-stationary data.
<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/ARIMA.png" alt="Streamlit Image 1" width="600" />
  
</div>
## Purpose:
- The ARIMA model is ideal for forecasting time series data, leveraging its own past values and past forecast errors. It's particularly effective for non-stationary data.

## Key Diagnostics:
### 1. Standardized Residuals:
- Display the deviations of actual values from predictions over time. Consistent patterns may indicate model inadequacies.

### 2. Residual Distribution:
- Histograms and density plots (KDE and normal distribution overlay) show how well residuals follow a normal distribution. A good fit has residuals closely following the normal curve.

### 3. Normal Q-Q Plots:
- Compare residual quantiles to a normal distribution. Points aligning with the reference line suggest normality in residuals. Deviations indicate potential issues.

#### 4. Model Diagnostics for Stocks:
- Specific diagnostics for the S&P 500 and Microsoft show the ARIMA model's performance, including residual plots and Q-Q plots, assessing model fit for these datasets.

## Conclusion:
These diagnostics are crucial for evaluating the ARIMA model's adequacy, ensuring reliable forecasts, and identifying areas for potential improvement.

#### 4. Random Forest Regression: 
Captures complex relationships via ensemble learning.

- A type of ensemble learning model that builds multiple decision trees and merges their results to get a more accurate and stable prediction. It’s great for capturing complex, non-linear relationships.

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/RandomForest.png" alt="Streamlit Image 1" width="600" />
  
</div>
## Mean Squared Error (MSE)
- **S&P 500:** The MSE is 113500.77, indicating a significant discrepancy between actual and predicted values.
- **Microsoft:** The MSE is 1102.30, showing better predictive accuracy compared to the S&P 500.

## Visual Analysis
- **S&P 500:** The actual values trend upwards, while the predicted values remain flat and lower, indicating poor model performance.
- **Microsoft:** The actual values also trend upwards with fluctuations, and the predicted values, although flatter, reflect some variability.


#### 5. LSTM (Long Short-Term Memory): 
Ideal for sequential and time-series data.

- LSTM networks are a type of recurrent neural network (RNN) capable of learning long-term dependencies. They are particularly well-suited for tasks where context over extended periods is crucial, such as language modeling, time series forecasting, and speech recognition.

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/LSTM.png" alt="Streamlit Image 1" width="600" />
  
</div>

## Visual Analysis
### 1. S&P 500 Prices Prediction:
- The LSTM model's predictions (red line) closely follow the actual prices (blue line), effectively capturing the upward trend and fluctuations.

### 2. Microsoft Prices Prediction:
- Similarly, for Microsoft, the predicted prices align well with the actual prices, reflecting the general trend and major movements accurately.

## Key Insights
### 1. Model Accuracy:
- The LSTM model provides accurate predictions, effectively capturing trends and fluctuations in both S&P 500 and Microsoft prices.

### 2. Performance Comparison:
- The LSTM model outperforms previous models (e.g., Random Forest) by closely aligning with actual market prices, showcasing its effectiveness for financial forecasting.

### 3. Future Improvements:
- Further accuracy can be achieved by fine-tuning hyperparameters, adding more training data, and incorporating additional features.

## Conclusion
The LSTM model demonstrates strong predictive capabilities for stock prices, making it a valuable tool for financial forecasting and informed investment decisions.


#### 6. SVM (Support Vector Machines):  
Handles non-linear relationships effectively.

- Originally designed for classification, but can be used in regression (SVR). It works well with non-linear relationships and can model complex relationships between the dependent variable and independent variables.
<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/SVM1.png" alt="Streamlit Image 1" width="600" />
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/SVM2.png" alt="Streamlit Image 1" width="600" />
  
</div>

## Insights

### Mean Squared Error (MSE)
- **S&P 500:** MSE is 19018.62, showing a notable discrepancy but better than previous models.
- **Microsoft:** MSE is 174.27, indicating high prediction accuracy.

### Visual Analysis
#### 1. S&P 500:
- Predictions (red line) closely follow actual prices (blue line), capturing trends but consistently underestimating values.

#### 2. Microsoft:
- Predictions align well with actual prices, effectively capturing trends and fluctuations with minor deviations.

### Key Insights
#### 1. Model Accuracy:
- The SVM model accurately predicts stock prices, especially for Microsoft, with lower MSE values than previous models.

#### 2. Performance Comparison:
- Performs better than the Random Forest model and competitively with the LSTM model, particularly for Microsoft.

#### 3. Future Improvements:
- Fine-tune hyperparameters and incorporate additional features for better accuracy.
- Regular updates with new data will maintain predictive power.

### Conclusion
The SVM model effectively predicts stock prices, especially for Microsoft, making it a valuable tool for financial forecasting and investment decisions.


#### 7. Principal Component Analysis (PCA): 
Reduces dimensionality and noise.

- Dimensionality reduction technique. It’s useful for simplifying the dataset, reducing noise, and identifying the most important variables that explain variability in your data, which can improve the performance of other models.

<div style="text-align: center;">
  <img src="https://github.com/shivakasireddy/UMBC-DATA606-SPRING2024-THURSDAY/blob/main/DOCS/PCA.png" alt="Streamlit Image 1" width="600" />
  
</div>

#### Analysis Summary:

1. **Correlation**: Microsoft shows a higher correlation with the S&P 500, indicating movements more aligned with the overall market, which could provide predictability in market trends.
2. **Beta**: Microsoft's higher Beta suggests it is more volatile, offering potentially higher risks and returns.
3. **Sharpe Ratio**: J.P. Morgan boasts a higher Sharpe Ratio, indicating superior risk-adjusted returns compared to Microsoft.

## Correlation Analysis

The correlation analysis between Microsoft and the S&P 500, and JP Morgan and the S&P 500 reveals the following insights:
- The correlation coefficient between Microsoft and the S&P 500 is 0.98. This indicates a very strong positive relationship, suggesting that Microsoft's stock price closely follows the market trends represented by the S&P 500.
- The correlation coefficient between JP Morgan and the S&P 500 is 0.74. Although this is also a strong positive relationship, it is less pronounced than that of Microsoft. This suggests that JP Morgan's stock price, while still following market trends, has more independent movement compared to Microsoft.

These high correlations indicate that both companies' stock prices are influenced significantly by the broader market movements.

## Beta Coefficient Analysis

The Beta Coefficient is a measure of a stock's volatility in relation to the market:

**Beta Coefficient:**
- Microsoft Beta: 1.21
- JP Morgan Beta: 1.12

Both stocks have betas above 1, indicating they are more volatile than the overall market.

- Microsoft's Beta is 1.21, indicating that it is 21% more volatile than the market. This suggests that when the market moves, Microsoft's stock price is likely to move in the same direction but with greater intensity.
- JP Morgan's Beta is 1.12, indicating that it is 12% more volatile than the market. This also implies a higher sensitivity to market movements but to a slightly lesser extent than Microsoft.

Both stocks having a Beta greater than 1 implies that they are more volatile compared to the overall market.

## Sharpe Ratio Analysis

**Sharpe Ratio:**
- Microsoft: 0.06
- JP Morgan: 0.56
- S&P 500: 0.52

The Sharpe Ratio comparison suggests that JP Morgan offers better risk-adjusted returns.

The Sharpe Ratio measures the risk-adjusted return of an investment:
- Microsoft's Sharpe Ratio is 0.06, which indicates a relatively low risk-adjusted return.
- JP Morgan's Sharpe Ratio is 0.56, which is significantly higher than Microsoft's, suggesting better risk-adjusted returns.
- The S&P 500 has a Sharpe Ratio of 0.52, which serves as a benchmark.

The comparison of Sharpe Ratios suggests that JP Morgan offers better returns for the level of risk taken compared to both Microsoft and the S&P 500.

The analysis indicates that Microsoft has a stronger correlation with market trends compared to JP Morgan. Both companies exhibit higher volatility than the market average, with Microsoft being the more volatile of the two. When considering risk-adjusted returns, JP Morgan stands out as offering superior performance relative to both Microsoft and the broader market. This makes JP Morgan a potentially more attractive investment from a risk-adjusted return perspective.


## Conclusion:
J.P. Morgan (JPM) is preferred for better risk-adjusted returns as per the Sharpe Ratio, but Microsoft might appeal to those seeking alignment with broader market movements.

## Benefits:
In financial data science and the stock market, these models can help:

- Forecast future stock prices or movements.
- Understand factors influencing price changes.
- Estimate the risk or volatility.
- Optimize investment portfolios.
- Detect and capitalize on market inefficiencies.
- Conduct algorithmic trading.





