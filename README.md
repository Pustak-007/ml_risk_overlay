# Project Charter

## 1. Objective & Central Hypothesis

**Objective:**  
To design, build, and rigorously test a machine learning-based risk management framework for short-volatility investment strategies. The goal is to create a systematic, data-driven signal that aims to preserve capital by avoiding catastrophic drawdowns historically associated with volatility "events."

**Central Hypothesis:**  
A logistic regression model, trained on a curated set of macro-financial indicators, can generate a forward-looking probability of a high-volatility regime. This probability can be used to systematically de-risk a portfolio, resulting in a superior risk-adjusted return profile compared to a naive, always-invested approach.

---

## 2. Feature Set (X)

The model will be trained on a parsimonious, theory-driven set of **nine features**, designed to capture distinct dimensions of market risk:

### Volatility Complex (Fear)
1. **VIX_level**: Spot VIX Index  
2. **Term_Structure_Ratio_1_0**: Ratio of 1st VIX Future to Spot VIX  
3. **Term_Structure_Slope_2_1**: Spread between 2nd and 1st VIX Futures  
4. **VVIX_level**: CBOE VVIX Index  

### Systemic Risk (Market Internals)
5. **Credit_Spread_High_Yield**: BofA US High Yield Index Spread  
6. **SPX_Realized_Vol_21d**: 21-day realized volatility of the S&P 500  
7. **Yield_Curve_Slope_10y_2y**: Spread between 10-Year and 2-Year US Treasury yields  

### Momentum (Rate of Change)
8. **VIX_Change_5d**: 5-day percentage change in the VIX  
9. **Credit_Spread_Change_21d**: 21-day percentage change in the Credit Spread  

---

## 3. Data & Scope

**Primary Data Sources:**  
- WRDS (CRSP, OptionMetrics)  
- Federal Reserve Economic Data (FRED)  

**Time Period:**  
January 1993 to the most recent available data. The effective start date of the final dataset will be determined by the feature with the latest inception date.

**Software:**  
Python 3.12.5, utilizing the `pandas`, `scikit-learn`, and `wrds` libraries.

---

## 4. Benchmark Strategy

- A "buy-and-hold" portfolio simulating a **-1.0x leveraged short-volatility product** (e.g., SVXY), rebalanced daily.  
- Represents the naive strategy without any risk management overlay.  

**Notes on Strategy Implementation:**  
- The ML-driven risk overlay will be applied on top of this risky product.  
- After Feb 2018 (Volmageddon), the daily inverse exposure to the S&P 500 VIX Short-Term Futures Index was reduced to **-0.5x** from **-1.0x**.  
- Our goal is to test how the risk overlay would have performed **if the leverage had remained at -1.0x** after Volmageddon.  
- Incidental analysis of the actual deleveraged product will also be conducted to compare performance.  
