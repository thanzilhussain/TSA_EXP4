# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 04-11-2026



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


data = pd.read_csv("user_behavior_timeseries.csv")
data.columns = data.columns.str.strip()

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data = data.groupby('Date').mean(numeric_only=True)

X = data["App Usage Time"]


trend = np.linspace(0, 100, len(X))
seasonal = 20 * np.sin(np.linspace(0, 3*np.pi, len(X)))
noise = np.random.normal(0, 5, len(X))

X = X + trend + seasonal + noise


plt.figure(figsize=(12,10))

# Original Data
plt.subplot(3,1,1)
plt.plot(X)
plt.title("Original Data - App Usage Time")
plt.xlabel("Time")
plt.ylabel("Usage")

# ACF
plt.subplot(3,1,2)
plot_acf(X, lags=15, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

# PACF
plt.subplot(3,1,3)
plot_pacf(X, lags=15, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.tight_layout()
plt.show()


model1 = ARIMA(X, order=(1,0,1)).fit()

phi = model1.params['ar.L1']
theta = model1.params['ma.L1']

ar = np.array([1, -phi])
ma = np.array([1, theta])

arma1 = ArmaProcess(ar, ma).generate_sample(nsample=500)

plt.figure(figsize=(12,10))

# ARMA Signal
plt.subplot(3,1,1)
plt.plot(arma1)
plt.title("Simulated ARMA(1,1) Process")
plt.xlabel("Time")
plt.ylabel("Value")

# PACF
plt.subplot(3,1,2)
plot_pacf(arma1, ax=plt.gca())
plt.title("PACF of ARMA(1,1)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

# ACF
plt.subplot(3,1,3)
plot_acf(arma1, ax=plt.gca())
plt.title("ACF of ARMA(1,1)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.tight_layout()
plt.show()

model2 = ARIMA(X, order=(2,0,2)).fit()

phi1 = model2.params['ar.L1']
phi2 = model2.params['ar.L2']
theta1 = model2.params['ma.L1']
theta2 = model2.params['ma.L2']

ar = np.array([1, -phi1, -phi2])
ma = np.array([1, theta1, theta2])

arma2 = ArmaProcess(ar, ma).generate_sample(nsample=500)

plt.figure(figsize=(12,10))

# ARMA Signal
plt.subplot(3,1,1)
plt.plot(arma2)
plt.title("Simulated ARMA(2,2) Process")
plt.xlabel("Time")
plt.ylabel("Value")

# PACF
plt.subplot(3,1,2)
plot_pacf(arma2, ax=plt.gca())
plt.title("PACF of ARMA(2,2)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

# ACF
plt.subplot(3,1,3)
plot_acf(arma2, ax=plt.gca())
plt.title("ACF of ARMA(2,2)")
plt.xlabel("Lag")
plt.ylabel("Correlation")

plt.tight_layout()
plt.show()
```
## OUTPUT:

<img width="1397" height="747" alt="image" src="https://github.com/user-attachments/assets/2694a010-8c55-48ac-99c0-c350b72c4e7d" />

<img width="1357" height="377" alt="image" src="https://github.com/user-attachments/assets/58b6b203-a2f9-4a8d-bdb5-4b2e30e3f7c9" />

<img width="1397" height="741" alt="image" src="https://github.com/user-attachments/assets/64113cd5-3f7c-4946-8f12-17ee99d00f53" />
<img width="1383" height="382" alt="image" src="https://github.com/user-attachments/assets/98946511-61de-45ec-8c23-6b3758a42daf" />
<img width="1362" height="743" alt="image" src="https://github.com/user-attachments/assets/46711ffb-ad5f-4aef-9aef-321780bc9ed3" />
<img width="1423" height="433" alt="image" src="https://github.com/user-attachments/assets/cde85a65-c3e7-49a8-a4ed-f769f941fc91" />

## RESULT:

Thus, a python program is created to fir ARMA Model successfully.
