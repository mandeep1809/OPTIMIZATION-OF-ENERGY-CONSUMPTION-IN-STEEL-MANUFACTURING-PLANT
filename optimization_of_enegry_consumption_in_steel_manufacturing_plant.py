'''

# Business Problem:

A leading steel manufacturer is facing challenges in accurately determining the most efficient steel grades based on energy 
consumption, production speed, and overall productivity. The absence of precise classification limits their ability to 
optimize manufacturing processes, streamline operations, and achieve cost savings.

**Objective:**

Maximize Product Efficiency.

**Constraint:**

Minimize the energy consumption and cycle time.

**Success Criteria:**

**Business Success Criteria:**
improve Production Efficiency by 20%

**ML Success Criteria:**

Achieve an accuracy of at least 95%..

**Economic Success Criteria:**

Achieve a cost saving of at least $1M


**CRISP-ML(Q) process model describes six phases:**

- Business and Data Understanding
- Data Preparation (Data Engineering)
- Model Building (Machine Learning)
- Model Evaluation and Tunning
- Deployment
- Monitoring and Maintenance

##  Data Collection

## Data Description :
This dataset contains  rows and columns, which are described as follows:

                                                                                    |
| Column Name            | Data Type  | Type of Data          | Scale   | Description   
|------------------------|-----------|----------------------|---------|---------------------------------------------  
| SRNO                  | INT       | Identifier           | -       | Serial number  
| DATETIME              | DATETIME  | Timestamp            | -       | Date and time of the data recorded  
| HEATNO                | INT       | Identifier           | -       | Unique identifier for the heat  
| GRADE                 | STRING    | Categorical          | -       | Grade of steel being processed  
| SECTION               | STRING    | Categorical          | -       | Section identifier  
| SEN_OPEN              | BOOLEAN   | Status               | -       | Sensor open status  
| HOT_COLD              | STRING    | Categorical          | -       | Indicates if the section is hot or cold  
| SECTION_IC            | BOOLEAN   | Status               | -       | Section's intermediate control status  
| SI_EAF                | STRING    | Categorical          | -       | Shift incharge name  
| COKE_REQ              | FLOAT     | Quantity             | MT      | Coke required amount  
| INJ1_QTY              | FLOAT     | Quantity             | MT      | Coke Injection Qty 1  
| INJ2_QTY              | FLOAT     | Quantity             | MT      | Coke Injection Qty 2  
| BSM                   | FLOAT     | Modifier             | -       | Blast furnace slag modifier  
| TP                    | FLOAT     | Chemical Composition | %       | Total phosphorus content  
| MSTB                  | FLOAT     | Balance Measurement  | -       | Metal slag through balance  
| SKULL                 | BOOLEAN   | Indicator            | -       | Skull formation indicator  
| SHRAD                 | FLOAT     | Content Measurement  | %       | Shredder content  
| REMET                 | FLOAT     | Content Measurement  | %       | Remet content  
| BP                    | FLOAT     | Power Measurement    | MW      | Blast furnace power  
| HBI                   | FLOAT     | Quantity             | MT      | Hot briquetted iron  
| OTHERS                | FLOAT     | Quantity             | MT      | Other materials quantity  
| SCRAP_QTY             | FLOAT     | Percentage           | %       | Scrap steel percentage  
| PIGIRON               | FLOAT     | Quantity             | MT      | Pig iron quantity  
| DRI1_QTY              | FLOAT     | Quantity             | MT      | DRI (Direct Reduced Iron) Lumps quantity  
| DRI2_QTY              | FLOAT     | Quantity             | MT      | DRI (Direct Reduced Iron) Fines quantity  
| TOT_DRI_QTY           | FLOAT     | Quantity             | MT      | Total DRI quantity  
| HOT_METAL             | FLOAT     | Quantity             | MT      | Hot metal from blast furnace  
| Total Charge          | FLOAT     | Quantity             | MT      | Total charge input  
| Hot_Heel              | FLOAT     | Quantity             | MT      | Leftover liquid metal in EAF  
| DOLO                  | FLOAT     | Quantity             | MT      | Dolomite content  
| DOLO1_EMPTY           | BOOLEAN   | Status               | -       | Dolomite 1 empty status  
| TOT_LIME_QTY          | FLOAT     | Quantity             | MT      | Total lime quantity  
| TAP_TEMP              | FLOAT     | Temperature          | °C      | Tapping temperature  
| O2REQ                 | FLOAT     | Quantity             | Nm³     | Oxygen required  
| O2ACT                 | FLOAT     | Quantity             | Nm³     | Actual oxygen content  
| ENERGY                | FLOAT     | Energy Consumption   | kWh     | Energy consumption  
| KWH_PER_TON           | FLOAT     | Energy Efficiency    | kWh/Ton | Energy consumption per ton  
| KWH_PER_MIN           | FLOAT     | Energy Efficiency    | kWh/min | Energy consumption per minute  
| MELT_TIME             | FLOAT     | Duration             | Min     | Melting time in EAF  
| TA_TIME               | FLOAT     | Duration             | Min     | Turnaround time  
| TT_TIME               | FLOAT     | Duration             | Min     | Total cycle time including breakdown  
| POW_ON_TIME           | FLOAT     | Duration             | Min     | Power on time  
| TAPPING_TIME          | FLOAT     | Duration             | Min     | Tapping time  
| ARCING_TIME           | FLOAT     | Duration             | Min     | Arcing time  
| DOWN_TIME             | FLOAT     | Duration             | Min     | Downtime  
| E1_CUR               | FLOAT     | Electrical Current   | A       | Electrode 1 current  
| E2_CUR               | FLOAT     | Electrical Current   | A       | Electrode 2 current  
| E3_CUR               | FLOAT     | Electrical Current   | A       | Electrode 3 current  
| SPOUT                | FLOAT     | Temperature          | °C      | Bottom refractory temperature (spout)  
| DOLOMIT              | FLOAT     | Temperature          | °C      | Dolomite temperature  
| CPC                  | FLOAT     | Quantity             | MT      | Calcined petroleum coke  
| TEMP_TIME            | FLOAT     | Duration             | Min     | Temperature time  
| TEMPERATURE          | FLOAT     | Temperature          | °C      | Temperature measurement  
| POWER                | FLOAT     | Power Measurement    | MW      | Power measurement  
| LAB_REP_TIME         | FLOAT     | Duration             | Min     | Laboratory reporting time  
| C                    | FLOAT     | Chemical Composition | %       | Carbon content  
| SI                   | FLOAT     | Chemical Composition | %       | Silicon content  
| MN                   | FLOAT     | Chemical Composition | %       | Manganese content  
| P                    | FLOAT     | Chemical Composition | %       | Phosphorus content  
| S                    | FLOAT     | Chemical Composition | %       | Sulfur content  
| CU                   | FLOAT     | Chemical Composition | %       | Copper content  
| CR                   | FLOAT     | Chemical Composition | %       | Chromium content  
| NI                   | FLOAT     | Chemical Composition | %       | Nickel content  
| N                    | FLOAT     | Chemical Composition | %       | Nitrogen content  
| TIME_UTLN_PRCNT      | FLOAT     | Utilization Rate     | %       | Time utilization percentage  
| TOP_COKE            | FLOAT     | Quantity             | MT      | Top coke usage  
| OPEN_C              | FLOAT     | Quantity             | MT      | Open coke usage  
| TAP_C               | FLOAT     | Quantity             | MT      | Tapping coke usage  
| IT_KG               | FLOAT     | Weight Measurement   | KG      | Iron tapping weight  
| BUCKET_NO           | INT       | Identifier           | -       | Bucket number identifier  
| STATIC_WT           | FLOAT     | Weight Measurement   | MT      | Static weight measurement  
| LIME                | FLOAT     | Quantity             | MT      | Lime quantity  
| LIME2               | FLOAT     | Quantity             | MT      | Lime 2 quantity  
| O2SIDE1             | FLOAT     | Quantity             | Nm³     | Oxygen side measurement 1  
| O2SIDE2             | FLOAT     | Quantity             | Nm³     | Oxygen side measurement 2  
| O2SIDE3             | FLOAT     | Quantity             | Nm³     | Oxygen side measurement 3  
| SPINNING            | BOOLEAN   | Indicator            | -       | Spinning indicator  
| RAMMING1            | BOOLEAN   | Indicator            | -       | Ramming 1 indicator  
| RAMMING2            | BOOLEAN   | Indicator            | -       | Ramming 2 indicator  
| PREV_TAP_TIME      | FLOAT     | Timestamp            | -       | Previous tapping time  
| GUNNING_NAME       | STRING    | Categorical          | -       | Gunning personnel name  
| SPINNING_NAME      | STRING    | Categorical          | -       | Spinning personnel name  
| RAMMING1_NAME      | STRING    | Categorical          | -       | Ramming 1 personnel name  
| RAMMING2_NAME      | STRING    | Categorical          | -       | Ramming 2 personnel name  
| TAP_DURATION       | FLOAT     | Duration             | Min     | Tapping duration  
| Pour_Back_Metal    | FLOAT     | Quantity             | MT      | Pour back metal quantity  
| LM_WT              | FLOAT     | Weight Measurement   | MT      | Liquid metal weight  
| Production         | FLOAT     | Quantity             | MT      | Production volume in metric tons  '''


------------------------------------- Importing necessary libraries---------------------------------------------------

import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For statistical data visualization
import sidetable  # For quick summary tables
from sklearn.compose import ColumnTransformer  # For column-wise transformations
from sklearn.pipeline import Pipeline  # For building pipelines
from sklearn.impute import SimpleImputer  # For imputing missing values
from sklearn.preprocessing import MinMaxScaler  # For scaling numerical features
from sklearn.preprocessing import OneHotEncoder  # For one-hot encoding categorical features
from feature_engine.outliers import Winsorizer  # For outlier treatment
from statsmodels.stats.outliers_influence import variance_inflation_factor  # For VIF calculation
from statsmodels.tools.tools import add_constant  # For adding constant to the model
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
import statsmodels.api as sm  # For statistical models and tests
from sklearn.linear_model import LinearRegression  # For linear regression modeling
from sklearn.metrics import r2_score  # For evaluating model performance
import joblib  # For saving and loading models
import pickle  # For serializing and deserializing Python objects
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV  # For cross-validation and hyperparameter tuning
from sklearn.feature_selection import RFE  # For recursive feature elimination
from sqlalchemy import create_engine  # For database connection
from urllib.parse import quote
import warnings  # Suppress warnings
warnings.filterwarnings("ignore")

---------------------------------------    Load Dataset    ---------------------------------------------------------------

df_clean = pd.read_csv(r"E:/Mandeep/360 DigiTMG/PROJECTS/OPTIMIZATION STEEL MANUFACTURING(3RD)/DATA SETS/df_clean_Manufacturing.csv")

# Credentials to connect to Database
user = 'root'  # user name
pw = quote('Mandeep@1809')  # password
db = 'Steel'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# to_sql() - function to push the dataframe onto a SQL table.
df_clean.to_sql('manufacture', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

### To read the data from MySQL Database
sql = 'select * from manufacture;'

# Read data from SQL database into a DataFrame
df = pd.read_sql_query(sql, engine) 
# Use 'pd.read_sql_query' to execute the SQL query and read the result into a pandas DataFrame.
# 'sql' is the SQL query string, and 'engine' is the SQLAlchemy engine object created earlier.

# Displaying the DataFrame
print(df)



"""# Initially we had  Rows &   Columns


"""

df.shape

df.columns

"""# Data Cleaning Steps"""

# Set the correct headers by skipping the first two rows
df_clean = pd.read_csv('/content/steel_manufacturing.csv', skiprows=2)

df_clean.columns

df_clean.shape

from google.colab import files
csv_filename = 'df_clean_Manufacturing.csv'
df_clean.to_csv(csv_filename, index=False)
files.download(csv_filename)

df_clean.head()

# 1. Handling Missing Values
df_clean.info() #no any missing value found in any columns

df_clean["PREV_TAP_TIME"].isnull().sum()

df_clean["Production (MT)"].isnull().sum()

df_sorted = df_clean.sort_values(by=['SRNO'], ascending=[True])
print(df_sorted)

"""# 1. TYPE CASTING"""

print(df_clean.dtypes)

df_clean['DATETIME'] = pd.to_datetime(df_clean['DATETIME'], errors='coerce')

num_cols = df_clean.select_dtypes(include=['object']).columns.difference(['DATETIME', 'GRADE', 'SECTION_IC'])

print(num_cols)

df_clean[num_cols] = df_clean[num_cols].apply(pd.to_numeric, errors='coerce')

print(df_clean.dtypes)

df_clean.info()

"""# 2. HANDLING DUPLICATES"""

df_clean.drop_duplicates(inplace=True)

"""# 3. OUTLIER ANALYSIS (Using IQR Method)"""

# Convert relevant columns to numeric type
for column in df_clean.select_dtypes(include=['object']).columns:
    try:
        # Attempt to convert the column to numeric
        df_clean[column] = pd.to_numeric(df_clean[column], errors='coerce')  # 'coerce' will set invalid values to NaN
    except ValueError:
        print(f"Could not convert column '{column}' to numeric. Check for non-numeric values.")

# Calculate quantiles and IQR for numerical columns only
Q1 = df_clean.select_dtypes(include=np.number).quantile(0.25)
Q3 = df_clean.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

# Detect and remove outliers
# This needs to be applied only to the numerical columns
numerical_df = df_clean.select_dtypes(include=np.number)
outliers = ((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).sum()
print("Outliers Count:", outliers[outliers > 0])

# Remove outliers based on numerical columns only
df_clean = df_clean[~((numerical_df < (Q1 - 1.5 * IQR)) | (numerical_df > (Q3 + 1.5 * IQR))).any(axis=1)]

"""# 4. ZERO & NEAR ZERO VARIANCE FEATURES"""

from sklearn.feature_selection import VarianceThreshold
var_threshold = VarianceThreshold(threshold=0.01)
df_clean_var = df_clean.drop(columns=['DATETIME', 'GRADE', 'SECTION_IC'])
df_clean_var = df_clean_var.loc[:, var_threshold.fit(df_clean_var).get_support()]
print("Remaining Features after Zero Variance Filtering:", df_clean_var.columns)
df_clean = df_clean[list(df_clean_var.columns) + ['DATETIME', 'GRADE', 'SECTION_IC']]

"""# 5. HANDLING MISSING VALUES"""

df_clean.fillna(method='ffill', inplace=True)
print("Missing Values After Handling:")
print(df_clean.isnull().sum().sum())

"""# 6. DISCRETIZATION/BINNING/GROUPING (Example: Categorizing Energy Consumption)"""

df_clean['ENERGY (Energy Consumption)'] = pd.qcut(df_clean['ENERGY (Energy Consumption)'], q=4, labels=["Low", "Medium", "High", "Very High"])

"""# 7. DUMMY VARIABLE CREATION"""

df_clean = pd.get_dummies(df_clean, columns=['GRADE', 'SECTION_IC'], drop_first=True)

"""# 8. TRANSFORMATION (Scaling Numeric Data)"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_cols = df_clean.select_dtypes(include=['number']).columns.difference(['SRNO', 'HEATNO'])
df_clean[scaled_cols] = scaler.fit_transform(df_clean[scaled_cols])

"""# Save Cleaned Data"""

df_clean.to_csv("cleaned_data.csv", index=False)