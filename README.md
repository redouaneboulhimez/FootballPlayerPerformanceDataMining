# Football Player Performance Analysis - Complete Project Analysis

## 📊 Project Overview

This project implements a **regression-based data mining analysis** to predict football player performance, specifically **Goals Per 90 minutes**, using advanced football statistics and match data. The project follows a complete data mining workflow from data exploration to model evaluation and visualization.

---

## 🎯 Problem Statement

**Objective**: Predict a player's **Goals Per 90 minutes** (goals scored per 90 minutes of playing time) using match statistics and advanced analytics metrics.

**Business Value**: 
- Identify key performance indicators that drive goal-scoring
- Understand which player attributes correlate with high goal-scoring rates
- Provide insights for player evaluation and recruitment

---

## 📁 Dataset Information

### Source
- **Dataset**: FBref Football Player Performance Dataset
- **File**: `PlayersFBREF.csv`
- **Total Records**: 2,274 players
- **Total Features**: 34 columns
- **Data Quality**: No missing values (clean dataset)

### Available Features
The dataset includes:
- **Basic Info**: Player name, Nation, Position, Age
- **Playing Time**: Matches Played, Starts, Minutes, 90s Played
- **Goal Statistics**: Goals, Assists, Goals + Assists, Non-Penalty Goals, Penalty Goals
- **Advanced Metrics**: xG (Expected Goals), npxG (Non-Penalty xG), xAG (Expected Assists)
- **Progressive Actions**: Progressive Carries, Progressive Passes, Progressive Receives
- **Per-90 Metrics**: All metrics normalized to per-90-minute rates
- **Disciplinary**: Yellow Cards, Red Cards

---

## 🔬 Methodology

### 1. Data Preprocessing

#### Feature Selection
Selected **10 features** for modeling:
- `Age` - Player age
- `90s Played` - Playing time (90-minute units)
- `xG Per 90` - Expected goals per 90 minutes
- `xAG Per 90` - Expected assists per 90 minutes
- `xG + xAG Per 90` - Combined expected goals and assists
- `npxG Per 90` - Non-penalty expected goals per 90
- `npxG + xAG Per 90` - Combined non-penalty xG and xAG
- `Progressive Carries` - Carries that advance the ball significantly
- `Progressive Passes` - Passes that advance the ball significantly
- `Progressive Receives` - Receiving the ball in advanced positions

**Rationale**: 
- Excluded player names, nations (not predictive)
- Focused on performance metrics and advanced analytics
- Included both raw totals and per-90 normalized metrics

#### Data Quality
- ✅ **No missing values** in selected features
- ✅ **No data cleaning required** (dataset was pre-cleaned)
- ✅ **Appropriate data types** (numeric features for regression)

### 2. Data Splitting

- **Train/Test Split**: 80/20 (1,819 training / 455 testing samples)
- **Random State**: 42 (for reproducibility)
- **Stratification**: Not used (regression task, continuous target)

### 3. Feature Scaling

- **StandardScaler** applied to Linear Regression features
- **Rationale**: Linear models benefit from normalized features
- **Note**: Random Forest does not require scaling (tree-based)

### 4. Models Implemented

#### Model 1: Linear Regression
- **Type**: Ordinary Least Squares (OLS)
- **Scaling**: Yes (StandardScaler)
- **Advantages**: Interpretable coefficients, fast training
- **Use Case**: Baseline model, understanding linear relationships

#### Model 2: Random Forest Regressor
- **Type**: Ensemble of decision trees
- **Parameters**:
  - `n_estimators=200` (200 trees)
  - `max_depth=10` (prevents overfitting)
  - `random_state=42` (reproducibility)
  - `n_jobs=-1` (parallel processing)
- **Scaling**: No (tree-based, scale-invariant)
- **Advantages**: Captures non-linear relationships, feature importance
- **Use Case**: Advanced model, handling complex patterns

#### Model 3: XGBoost Regressor
- **Type**: Gradient Boosting framework
- **Parameters**:
  - `n_estimators=200` (200 boosting rounds)
  - `max_depth=6` (tree depth)
  - `learning_rate=0.1` (shrinkage)
  - `random_state=42` (reproducibility)
  - `n_jobs=-1` (parallel processing)
- **Scaling**: No (tree-based, scale-invariant)
- **Advantages**: State-of-the-art performance, handles missing values, regularization
- **Use Case**: High-performance model for tabular data
- **Note**: Falls back to GradientBoostingRegressor if XGBoost library not installed

#### Model 4: Elastic Net
- **Type**: Regularized linear regression (L1 + L2)
- **Parameters**:
  - `alpha=1.0` (regularization strength)
  - `l1_ratio=0.5` (balance between L1 and L2)
  - `random_state=42` (reproducibility)
- **Scaling**: Yes (StandardScaler)
- **Advantages**: Handles multicollinearity, feature selection, robust to outliers
- **Use Case**: Addresses multicollinearity issues in Linear Regression

---

## 📈 Results & Performance Analysis

### Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **Linear Regression** | **0.074** | **0.175** | **0.368** |
| Random Forest | 0.068 | 0.185 | 0.289 |
| XGBoost | 0.070 | 0.187 | 0.278 |
| Elastic Net | 0.123 | 0.220 | -0.003 |

### Performance Interpretation

#### Linear Regression (Best Overall)
- **MAE (0.074)**: Average prediction error of 0.074 goals per 90 minutes
  - Example: If actual = 0.5 goals/90, prediction typically within 0.43-0.57
- **RMSE (0.175)**: Penalizes larger errors; indicates some outliers
- **R² (0.368)**: Explains **36.8%** of variance in Goals Per 90
  - Moderate explanatory power (typical for sports analytics)
  - Remaining 63.2% variance due to unmeasured factors

#### Random Forest
- **Best MAE** (0.068): Slightly better average accuracy than Linear Regression
- **Worse RMSE** (0.185 vs 0.175): More large errors (overfitting risk)
- **Lower R²** (0.289 vs 0.368): Explains less variance
  - Possible overfitting or insufficient tuning

#### XGBoost
- **MAE (0.070)**: Good average accuracy, between Linear Regression and Random Forest
- **RMSE (0.187)**: Similar to Random Forest, indicates some large errors
- **R² (0.278)**: Lower than Linear Regression, similar to Random Forest
  - May benefit from hyperparameter tuning
  - Gradient boosting approach shows promise

#### Elastic Net
- **MAE (0.123)**: Higher error than other models
- **RMSE (0.220)**: Highest error among all models
- **R² (-0.003)**: Negative R² indicates model performs worse than baseline
  - Current hyperparameters (alpha=1.0, l1_ratio=0.5) may be too aggressive
  - Requires tuning to find optimal regularization strength
  - Despite poor performance, useful for addressing multicollinearity

### Conclusion
**Linear Regression performs best overall** - highest R² (0.368) and best RMSE (0.175) indicate it generalizes best to unseen data. Random Forest has the best MAE (0.068) but struggles with variance explanation. XGBoost shows competitive performance and may improve with tuning. Elastic Net requires hyperparameter optimization to be effective.

---

## 🔍 Feature Analysis

### Linear Regression Coefficients

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `npxG + xAG Per 90` | **+0.450** | Strongest positive predictor |
| `xG Per 90` | **+0.372** | Second strongest predictor |
| `Progressive Receives` | **+0.066** | Moderate positive impact |
| `90s Played` | **+0.009** | Minimal positive impact |
| `Age` | **+0.001** | Negligible impact |
| `Progressive Passes` | **-0.005** | Minimal negative impact |
| `Progressive Carries` | **-0.029** | Small negative impact |
| `xAG Per 90` | **-0.134** | Moderate negative (multicollinearity) |
| `xG + xAG Per 90` | **-0.187** | Negative (redundant with other features) |
| `npxG Per 90` | **-0.480** | Strong negative (multicollinearity issue) |

**Key Insights**:
- **Expected goals metrics** are the strongest predictors
- **Multicollinearity** present (multiple xG features are correlated)
- **Progressive Receives** positively impacts goal-scoring
- **Age** has minimal impact (surprising but data-driven)

### Random Forest Feature Importance

| Feature | Importance | Rank |
|---------|------------|------|
| `xG Per 90` | **60.8%** | 🥇 Most Important |
| `90s Played` | **10.8%** | 🥈 Second |
| `Progressive Receives` | **6.5%** | 🥉 Third |
| `Progressive Carries` | 3.7% | 4th |
| `Progressive Passes` | 3.7% | 5th |
| `npxG Per 90` | 3.7% | 6th |
| `xG + xAG Per 90` | 3.4% | 7th |
| `xAG Per 90` | 3.1% | 8th |
| `Age` | 2.3% | 9th |
| `npxG + xAG Per 90` | 2.1% | 10th |

**Key Insights**:
- **xG Per 90 dominates** (60.8% importance) - single most important feature
- **Playing time matters** (10.8%) - more minutes = more opportunities
- **Progressive actions** collectively important (~14% combined)
- **Age is least important** (2.3%) - confirms minimal impact

---

## 📊 Visualizations Analysis

### 1. Predicted vs Actual Scatter Plots
- **Purpose**: Visual assessment of prediction accuracy for all models
- **Layout**: 2x2 grid displaying all 4 models (Linear Regression, Random Forest, XGBoost, Elastic Net)
- **Interpretation**: 
  - Points closer to red diagonal line = better predictions
  - All models show reasonable fit with some scatter
  - Linear Regression shows tightest clustering around diagonal
  - Elastic Net shows more dispersion (reflecting lower performance)

### 2. Residual Plots
- **Purpose**: Check for prediction bias and patterns across all models
- **Layout**: 2x2 grid displaying residuals for all 4 models
- **Interpretation**:
  - Random scatter around zero = good (no systematic bias)
  - Patterns (curves, clusters) = model missing relationships
  - Linear Regression and Random Forest show relatively random residuals
  - XGBoost residuals similar to Random Forest
  - Elastic Net shows more systematic patterns (needs tuning)

### 3. Feature Importance Bar Chart
- **Purpose**: Visualize which features Random Forest considers most important
- **Interpretation**: Clear dominance of xG Per 90 feature (60.8% importance)

---

## ✅ Project Strengths

1. **Clean Workflow**: Well-structured, step-by-step approach
2. **Proper Methodology**: Train/test split, scaling, multiple models
3. **Comprehensive Evaluation**: Multiple metrics (MAE, RMSE, R²)
4. **Good Visualizations**: Scatter plots, residuals, feature importance
5. **Interpretable Results**: Clear insights from coefficients and importance
6. **Reproducible**: Random state set, code is clear
7. **Data Quality**: Clean dataset with no missing values
8. **Feature Engineering**: Good selection of relevant football metrics

---

## ⚠️ Limitations & Challenges

### 1. Model Performance
- **R² = 0.368**: Only explains ~37% of variance
  - **Reason**: Many unmeasured factors (opponent quality, form, tactics, luck)
  - **Acceptable**: Common in sports analytics (high variance domain)

### 2. Multicollinearity
- **Issue**: Multiple xG-related features are highly correlated
  - `xG Per 90`, `npxG Per 90`, `xG + xAG Per 90`, `npxG + xAG Per 90`
- **Impact**: 
  - Unstable coefficients in Linear Regression
  - Negative coefficients for some xG features (counterintuitive)
- **Solution**: Feature selection or dimensionality reduction (PCA)

### 3. Feature Selection
- **Missing Context**: No opponent strength, team quality, league difficulty
- **No Temporal Data**: No form, recent performance trends
- **No Positional Context**: Position not used (could be important)

### 4. Random Forest Underperformance
- **Lower R² than Linear Regression**: Unusual (RF usually better)
- **Possible Causes**:
  - Overfitting to training data
  - Insufficient tuning (hyperparameters)
  - Linear relationships dominate (RF advantage not utilized)

### 5. Data Limitations
- **Single Season**: May not capture long-term patterns
- **No Match-Level Data**: Only aggregated season totals
- **No Contextual Factors**: Weather, home/away, match importance

---

## 🚀 Recommendations for Improvement

### Short-Term Improvements

1. **Feature Selection**
   - Remove redundant xG features (keep only `xG Per 90` and `xAG Per 90`)
   - Add position as categorical feature (one-hot encoding)
   - Create interaction features (e.g., `xG Per 90 × Progressive Receives`)

2. **Model Tuning**
   - **Random Forest**: Tune `max_depth`, `min_samples_split`, `n_estimators`
   - Try **Gradient Boosting** (XGBoost, LightGBM) - often better than RF
   - Try **Ridge/Lasso Regression** - handles multicollinearity better

3. **Feature Engineering**
   - Create efficiency ratios: `Goals / xG` (finishing quality)
   - Normalize progressive actions by position
   - Add per-90 versions of progressive actions

### Medium-Term Enhancements

4. **Additional Models** ✅ **IMPLEMENTED**
   - ✅ **XGBoost Regressor**: State-of-the-art for tabular data - **Now implemented**
   - ✅ **Elastic Net**: Combines Ridge + Lasso (handles multicollinearity) - **Now implemented**
   - **Neural Network**: If more data available (future enhancement)

5. **Advanced Evaluation**
   - **Cross-Validation**: 5-fold or 10-fold CV for robust metrics
   - **Learning Curves**: Check for overfitting/underfitting
   - **Feature Importance Comparison**: Compare across all models

6. **Data Enhancement**
   - Add league/competition level
   - Include team strength metrics
   - Add recent form (last 5-10 matches)

### Long-Term Extensions

7. **Alternative Targets**
   - Predict `Assists Per 90`
   - Predict `Goals + Assists Per 90` (combined metric)
   - Predict `Non-Penalty Goals Per 90`

8. **Classification Task**
   - Convert to classification: "High Performer" (Goals/90 > 0.5) vs "Low Performer"
   - Use Logistic Regression, Random Forest Classifier
   - Evaluate with precision, recall, F1-score

9. **Time Series Analysis**
   - If match-level data available: predict next match performance
   - Include recent form, momentum features

---

## 📝 Code Quality Assessment

### Strengths
- ✅ Clear variable naming
- ✅ Proper imports and organization
- ✅ Comments explaining steps
- ✅ Reproducible (random_state set)
- ✅ Good use of pandas and sklearn

### Areas for Improvement
- ⚠️ Could add more comments explaining *why* certain choices
- ⚠️ Could modularize into functions (reusability)
- ⚠️ Could add error handling
- ⚠️ Could save models for future use (pickle/joblib)

---

## 🎓 Learning Outcomes

This project demonstrates:
1. ✅ Complete data mining workflow (CRISP-DM methodology)
2. ✅ Regression modeling techniques
3. ✅ Model evaluation and comparison
4. ✅ Feature importance analysis
5. ✅ Data visualization for model diagnostics
6. ✅ Sports analytics application

---

## 📚 Technical Stack

- **Language**: Python 3.x
- **Libraries**:
  - `pandas`: Data manipulation
  - `numpy`: Numerical operations
  - `scikit-learn`: Machine learning models (Linear Regression, Random Forest, Elastic Net)
  - `xgboost`: XGBoost Regressor (optional, falls back to GradientBoostingRegressor)
  - `matplotlib`: Basic plotting
  - `seaborn`: Statistical visualizations
- **Environment**: Jupyter Notebook
- **Data Format**: CSV

---

## 📊 Final Summary

### Key Findings
1. **xG Per 90 is the strongest predictor** of Goals Per 90 (60.8% importance in Random Forest)
2. **Linear Regression outperforms all other models** for this task (R² = 0.368)
3. **Model explains ~37% of variance** - moderate performance, typical for sports analytics
4. **Random Forest has best MAE** (0.068) but lower R², indicating overfitting risk
5. **XGBoost shows competitive performance** (R² = 0.278) and may improve with tuning
6. **Elastic Net requires hyperparameter optimization** to address multicollinearity effectively
7. **Progressive Receives** positively impacts goal-scoring
8. **Age has minimal impact** on goal-scoring rate

### Business Insights
- **Expected Goals (xG) is highly predictive** - validates advanced analytics
- **Playing time matters** - more minutes = more goal opportunities
- **Progressive actions** (especially receives) correlate with scoring
- **Simple linear model sufficient** - Linear Regression performs best, suggesting linear relationships dominate
- **Multiple model approach** - Comparing 4 models provides robust evaluation and insights

### Project Status
✅ **Complete and Functional** - All objectives achieved
- Data loaded and explored
- Features selected and prepared
- **4 models trained and evaluated** (Linear Regression, Random Forest, XGBoost, Elastic Net)
- Results visualized and interpreted (all models included in visualizations)
- Predictions saved for further analysis
- **Prediction functions support all 4 models** for flexible player evaluation

---

## 📁 Project Structure

```
DMProjet/
├── data/
│   ├── PlayersFBREF.csv          # Original dataset
│   └── predictions_results.csv    # Model predictions output
├── 01_football_player_performance.ipynb  # Main analysis notebook
└── untitled.md                    # This analysis document
```

---

## 🔗 Next Steps

1. **Hyperparameter Tuning**
   - Tune Elastic Net (alpha, l1_ratio) to improve performance
   - Optimize XGBoost parameters (learning_rate, max_depth, n_estimators)
   - Use GridSearchCV or RandomizedSearchCV for systematic tuning

2. **Feature Selection**
   - Remove redundant xG features to address multicollinearity
   - Test model performance with reduced feature set

3. **Extend to other targets** (Assists, Combined metrics)
4. **Add more data** (opponent strength, team quality)
5. **Deploy model** (if needed for production use)

---

**Project Completion Date**: Analysis completed  
**Status**: ✅ Complete - Ready for presentation or further development
