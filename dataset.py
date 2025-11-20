from pathlib import Path
from typing import Optional, Union
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# dataset.py


def load_clean_dataset(path: Optional[Union[str, Path]] = None, **read_csv_kwargs) -> pd.DataFrame:
    """
    Load Clean_Dataset.csv into a pandas DataFrame.

    If path is None, this function will look for 'Clean_Dataset.csv' in the same
    directory as this file.

    Additional keyword arguments are passed to pandas.read_csv.
    """
    if path is None:
        path = Path(__file__).parent / "Clean_Dataset.csv"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # sensible defaults, allow override via read_csv_kwargs
    defaults = {"encoding": "utf-8", "low_memory": False}
    defaults.update(read_csv_kwargs)

    df = pd.read_csv(path, **defaults)
    return df
def liner_regression(df):
    '''Perform linear regression on the dataset.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
    Returns:
        model (LinearRegression): The trained linear regression model.
        mse (float): Mean Squared Error of the model on the test set.
        r2 (float): R^2 score of the model on the test set.
    '''
    x = df[['days_left']]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2
def polynomial_regression(df, degree=2):
    '''Perform polynomial regression on the dataset.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.
        degree (int): The degree of the polynomial features.
    Returns:
        model (Pipeline): The trained polynomial regression model.
        mse (float): Mean Squared Error of the model on the test set.
        r2 (float): R^2 score of the model on the test set.
    '''
    x = df[['days_left']]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

if __name__ == "__main__":
    df = load_clean_dataset(path="./dataset/Clean_Dataset.csv")
    df_eco = df[df['class'] == 'Economy']
    df_eco = df_eco.drop(columns=['class'])
    df_bus = df[df['class'] == 'Business']
    df_bus = df_bus.drop(columns=['class'])
    model_eco, mse_eco, r2_eco = liner_regression(df_eco)
    model_bus, mse_bus, r2_bus = liner_regression(df_bus)
    print(f"Mean Squared Error of Economy Class: {mse_eco}")
    print(f"R^2 Score of Economy Class: {r2_eco}")
    print(f"Mean Squared Error of Business Class: {mse_bus}")
    print(f"R^2 Score of Business Class: {r2_bus}")   
    '''
    Mean Squared Error: 9425177.940393236
    R^2 Score: 0.311626645114617
    Mean Squared Error: 167576174.98983854
    R^2 Score: 0.0090904878457857
    From the results, we can see that the linear regression model performs better on the Economy class dataset compared to the Business class dataset.
    The R^2 score for the Economy class is significantly higher, indicating that the model explains a larger proportion of the variance in prices for Economy class tickets.
    In contrast, the Business class model has a very low R^2 score, suggesting that the linear regression model is not a good fit for predicting prices in this category.
    However, both models have relatively high Mean Squared Errors, indicating that there is still considerable error in the predictions.
    '''
    model_eco_poly, mse_eco_poly, r2_eco_poly = polynomial_regression(df_eco, degree=3)
    model_bus_poly, mse_bus_poly, r2_bus_poly = polynomial_regression(df_bus, degree=3)
    print(f"Mean Squared Error of Economy Class (Polynomial): {mse_eco_poly}")
    print(f"R^2 Score of Economy Class (Polynomial): {r2_eco_poly}")
    print(f"Mean Squared Error of Business Class (Polynomial): {mse_bus_poly}")
    print(f"R^2 Score of Business Class (Polynomial): {r2_bus_poly}")   
    '''
    Mean Squared Error of Economy Class (Polynomial): 7931503.535705547
    R^2 Score of Economy Class (Polynomial): 0.4207180243505184
    Mean Squared Error of Business Class (Polynomial): 166030986.77541304
    R^2 Score of Business Class (Polynomial): 0.018227477037928708
    The polynomial regression model shows an improvement over the linear regression model for both Economy and Business class datasets.
    However, the improvement is more pronounced in the Economy class, where the R^2 score increased significantly, indicating a better fit.
    The Business class model also shows a slight improvement in R^2 score, but it remains relatively low, suggesting that even with polynomial features, the model struggles to capture the underlying patterns in the data for Business class tickets.
    Overall, while polynomial regression provides better performance, especially for Economy class, there is still room for improvement in modeling ticket prices, particularly for Business class.
    '''