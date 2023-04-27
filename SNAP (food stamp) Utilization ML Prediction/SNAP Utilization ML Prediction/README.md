### SNAP Eligibility Jupyter Notebook <br>
[Data Source]() Notebook containing the Python code for the modeling process of this target variable

[Data Source]() __1_household-codebook-puf__: Official FoodAPS explanation and guide about the survey data collection process, variables included, and data type description for each variable

### Data Files <br>
[Data Source]() __faps_household_puf (2)__: public dataset of the FoodAPS survey conducted by ERS and FNS

__SNAPHH_column_reference.csv__: Manually created document for the purpose of data variable organization. This file contains columns acting as lists that organizes each variable in the dataset according to it's topic or theme. For example, separating the variables by
    * socio-economic variables
    * nutrition variables
    * SNAP targeting data
    * administrative survey data

__data_processing.py__: Manually created Python package with functions integral to the processing of data as stated in the SNAPHH_column_reference document. Functions include:<br>
    - Data Cleaning<br>
    - Interactive Null Value Detection<br>
    - Preliminary Decision Tree Feature Selection<br>

__model.py__: Manually created Python package with functions created to automate and replicate the model creation portions of the project. Functions include: <br>
    - Logistic Model Processing and Evaluation <br>
    - Linear Model GirdSearch Exploration <br>
    - XGBoost GridSearch Exploration <br>
    - Graphing for Threshold Tampering <br>

__SNAP_eligibility_XGB_RF_model_results__ : The results of the gridsearch for the linear models. This file has been pickled from the code to avoid repetative processing at long run times

__SNAP_eligibility_linear_model_results__: The results of the gridsearch for the XGBoost model. This file has been pickled from the code to avoid repetative processing at long run times
