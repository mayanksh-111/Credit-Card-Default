# Credit Card Default Prediction Project Documentation

## Project Overview

This project implements a comprehensive machine learning approach to predict credit card default risk using a dataset of Taiwanese credit card customers. The project demonstrates the effectiveness of various machine learning algorithms and explores feature engineering techniques to improve model performance and computational efficiency.

**Primary Objective**: Develop an accurate and efficient model to predict the likelihood of credit card default for risk management purposes.

**Key Innovation**: Implementation of feature reduction techniques using Principal Component Analysis (PCA) that reduced runtime by approximately 60% while maintaining model accuracy.

## Dataset Description

The dataset contains credit card customer information from Taiwan with the following characteristics:

- **Size**: 30,000 customer records
- **Features**: 27 variables including:
  - **Demographic Information**: Customer_ID, marriage, sex, education, age
  - **Credit Information**: LIMIT_BAL (credit limit)
  - **Payment History**: pay_0, pay_2, pay_3, pay_4, pay_5, pay_6 (payment status over 6 months)
  - **Bill Amounts**: Bill_amt1 through Bill_amt6 (bill statements for 6 months)
  - **Payment Amounts**: pay_amt1 through pay_amt6 (payment amounts for 6 months)
  - **Engineered Features**: AVG_Bill_amt, PAY_TO_BILL_ratio
  - **Target Variable**: next_month_default (binary: 0 = no default, 1 = default)

## Methodology

### Phase 1: Initial Model Comparison (80-20 Split)

The project employed a systematic approach to compare multiple machine learning algorithms:

#### Data Preprocessing
- **Data Cleaning**: Handled missing values using SimpleImputer
- **Feature Scaling**: Applied StandardScaler for algorithms requiring normalization
- **Train-Test Split**: 80% training, 20% testing with random state for reproducibility

#### Model Implementations and Accuracies

| Model | Implementation | Accuracy | Notes |
|-------|---------------|----------|-------|
| **Linear Regression** | Manual implementation with bias term | [Space for accuracy] | Baseline linear model |
| **Ridge Regression** | Manual with cross-validation | [Space for accuracy] | Regularized linear model |
| **Decision Tree** | Custom implementation | [Space for accuracy] | Max depth = 5 |
| **Random Forest (Custom)** | Custom implementation | [Space for accuracy] | n_estimators=10, max_depth=5 |
| **Random Forest (Sklearn)** | sklearn.RandomForestClassifier | [Space for accuracy] | n_estimators=100, max_depth=5 |
| **LightGBM** | lgb.LGBMClassifier | [Space for accuracy] | n_estimators=120, max_depth=5 |
| **XGBoost** | xgb.XGBClassifier | [Space for accuracy] | n_estimators=90, max_depth=5 |

**Best Performing Model**: Random Forest (Sklearn implementation) achieved the highest accuracy among all tested models.

### Phase 2: Feature Analysis and Selection

#### Initial Feature Importance Analysis
The project identified weak correlations between certain demographic features and the target variable:
- **Low Correlation Features**: marriage, sex, education, age, PAY_TO_BILL_ratio, AVG_Bill_amt
- These features showed minimal predictive power for default prediction

#### Payment History Feature Correlation
Analysis revealed high correlation among payment history features:
- **Highly Correlated Features**: pay_0, pay_2, pay_3, pay_4, pay_5, pay_6
- These features represent payment status across different months and show strong interdependence

### Phase 3: Feature Engineering and Dimensionality Reduction

#### Principal Component Analysis (PCA) Implementation
- **Target Features**: Combined pay_0, pay_2, pay_3, pay_4, pay_5, pay_6 using PCA
- **Dimensionality Reduction**: Reduced 6 payment features to 1 principal component (PAY_PCA)
- **Standardization**: Applied StandardScaler before PCA transformation
- **Variance Retained**: Single component captured majority of variance from payment features

#### Feature Selection Strategy
1. **Excluded Features**: 
   - Low-importance demographic features: marriage, sex, education, age
   - Redundant features: PAY_TO_BILL_ratio, AVG_Bill_amt
   - Individual bill amounts: Bill_amt1 through Bill_amt6
   - Original payment features: pay_0, pay_2, pay_3, pay_4, pay_5, pay_6

2. **Retained Features**:
   - Customer_ID (for identification)
   - LIMIT_BAL (credit limit)
   - Individual payment amounts: pay_amt1 through pay_amt6
   - PAY_PCA (principal component from payment history)

### Phase 4: Optimized Model Performance

#### Reduced Feature Set Results
| Model | Original Features | Reduced Features | Accuracy Improvement |
|-------|-------------------|------------------|---------------------|
| **Random Forest (Selected Features)** | 27 features | Selected subset | [Space for accuracy] |
| **Random Forest (PCA Features)** | 27 features | With PAY_PCA | [Space for accuracy] |
| **Custom Random Forest (PCA)** | 27 features | With PAY_PCA | [Space for accuracy] |

## Performance Improvements

### Computational Efficiency
- **Runtime Reduction**: Approximately 60% reduction in training time
- **Memory Usage**: Decreased due to reduced feature dimensionality
- **Model Complexity**: Simplified models with fewer parameters

### Model Interpretability
- **Feature Importance**: Clearer understanding of key predictive factors
- **Principal Component**: PAY_PCA represents consolidated payment behavior patterns
- **Simplified Architecture**: Easier to deploy and maintain in production

## Technical Implementation Details

### PCA Implementation
```python
# Standardization of payment features
scaler = StandardScaler()
pay_scaled = scaler.fit_transform(df[pay_columns])

# PCA transformation
pca = PCA(n_components=1)
df['PAY_PCA'] = pca.fit_transform(pay_scaled)
```

### Feature Correlation Analysis
- **Correlation Matrix**: Generated heatmap to visualize feature relationships
- **Target Correlation**: Identified features with strongest correlation to default risk
- **Multicollinearity**: Detected and addressed through PCA transformation

## Key Findings

### Model Performance Insights
1. **Random Forest Superiority**: Ensemble methods (Random Forest) consistently outperformed individual algorithms
2. **Feature Quality over Quantity**: Reduced feature set maintained predictive accuracy
3. **Payment History Importance**: Payment behavior patterns are critical predictors of default risk

### Feature Engineering Benefits
1. **Dimensionality Reduction**: PCA effectively condensed correlated payment features
2. **Computational Efficiency**: Significant reduction in training time without accuracy loss
3. **Noise Reduction**: Eliminated irrelevant demographic features that added noise

## Business Applications

### Risk Management
- **Early Warning System**: Identify potential defaulters before they miss payments
- **Credit Limit Optimization**: Adjust credit limits based on default probability
- **Portfolio Management**: Segment customers by risk levels for targeted interventions

### Operational Efficiency
- **Faster Decision Making**: Reduced model complexity enables real-time predictions
- **Resource Allocation**: Focus monitoring efforts on high-risk customers
- **Cost Reduction**: Automated risk assessment reduces manual review requirements

## Conclusions and Recommendations

### Model Selection
- **Recommended Algorithm**: Random Forest (Sklearn) due to superior accuracy and robustness
- **Feature Set**: Use PCA-reduced features for optimal performance-efficiency balance
- **Implementation**: Deploy with standardized preprocessing pipeline

### Future Enhancements
1. **Hyperparameter Tuning**: Optimize Random Forest parameters for specific dataset
2. **Cross-Validation**: Implement k-fold cross-validation for robust performance estimation
3. **Ensemble Methods**: Explore stacking or blending techniques for further improvements
4. **Real-Time Updates**: Implement online learning for model adaptation to new data

### Production Considerations
- **Model Monitoring**: Track performance metrics and feature drift
- **Scalability**: Ensure system can handle increased data volumes
- **Regulatory Compliance**: Maintain model explainability for regulatory requirements

## Technical Specifications

### Environment
- **Programming Language**: Python 3.x
- **Key Libraries**: scikit-learn, pandas, numpy, lightgbm, xgboost
- **Development Environment**: Jupyter Notebook
- **Hardware Requirements**: Standard CPU processing sufficient for training

### Reproducibility
- **Random State**: Set to 42 for consistent results
- **Version Control**: Track model versions and feature engineering steps
- **Documentation**: Comprehensive code comments and documentation

## Appendices

### Appendix A: Feature Descriptions
[Detailed description of all original features and their business meaning]

### Appendix B: Model Hyperparameters
[Complete parameter settings for all tested models]

### Appendix C: Performance Metrics
[Detailed accuracy, precision, recall, and F1-score results]

### Appendix D: Code Repository
[Link to complete implementation code and notebooks]

---

*This documentation provides a comprehensive overview of the credit card default prediction project, highlighting both the methodological approach and practical business applications. The successful implementation of PCA-based feature reduction demonstrates the importance of feature engineering in creating efficient and effective machine learning solutions.*