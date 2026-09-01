import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor, Lasso
from scipy import stats
import matplotlib.pyplot as plt

# Enable LaTeX text rendering globally
plt.rcParams['text.usetex'] = True
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\boldmath\bfseries' # or other packages that support bold

# Set the font family (e.g., to serif fonts often used with LaTeX)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] # Or other serif fonts

plt.rcParams.update({'font.size': 13}) # Default font size for most text

PANDAS_FOLDER_NAME = "market_data"

def convert_tick_data_to_pd(tick_files):
    """ Given the bid and ask data files, (1) create a pandas data frame and save it
    to the folder if it doesn't exist already
    """
    # read csv file
    data = []
    for tick_file in tick_files:
        curr_data = pd.DataFrame()
        curr_data = pd.read_csv(tick_file)
        curr_data.rename(columns={"UTC": "timestamp", "Bid": "bid", "Ask": "ask", "Bid Volume": "bid_volume", "Ask Volume": "ask_volume"}, inplace=True)
        data.append(curr_data)
    return data


def convert_bid_ask_data_to_pd(bid_files, ask_files, save_file_name):
    """ Given the bid and ask data files, (1) create a pandas data frame and save it
    to the folder if it doesn't exist already
    """
    save_file_path = os.path.join(PANDAS_FOLDER_NAME, save_file_name)
    if os.path.exists(os.path.join(PANDAS_FOLDER_NAME, save_file_name)):
        data = pd.read_pickle(save_file_path)
        return data

    # read csv file
    data = pd.DataFrame()
    for bid_file, ask_file in zip(bid_files, ask_files):
        bid_data = pd.read_csv(bid_file)
        ask_data = pd.read_csv(ask_file)

        bid_data.rename(columns={"UTC": "timestamp", "Open": "bid_open", "Close": "bid_close", "High": "bid_high", "Low": "bid_low", "Volume": "bid_volume"}, inplace=True)
        ask_data.rename(columns={"UTC": "timestamp", "Open": "ask_open", "Close": "ask_close", "High": "ask_high", "Low": "ask_low", "Volume": "ask_volume"}, inplace=True)

        # merge bid and ask data
        curr_data = pd.merge(bid_data, ask_data, on="timestamp", how='left')
        data = pd.concat([data, curr_data])

    data.to_pickle(save_file_path)
    return data


def plot_volume_price_chart(data, start=None, end=None):
    """
    Parameters:
    -----------
    data : pd.DataFrame
        The dataframe containing bid-ask data
    title : str
        Title for the plot
    start : int, optional
        Starting index for the data slice (default: None, which means 0)
    end : int, optional
        Ending index for the data slice (default: None, which means len(data))
    """
    # Slice the data based on start and end indices
    if start is None:
        start = 0
    if end is None:
        end = len(data)

    data_slice = data.iloc[start:end]

    fig, axes = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

    # Top: Price chart with ranges
    # Plot bid range (low to high)
    axes.fill_between(data_slice['timestamp'], data_slice['bid_low'], data_slice['bid_high'],
                          alpha=0.2, color='blue', label='Bid Range (Low-High)')
    axes.plot(data_slice['timestamp'], data_slice['bid_close'], label='Bid Close', color='darkblue', linewidth=1.5)

    # Plot ask range (low to high)
    axes.fill_between(data_slice['timestamp'], data_slice['ask_low'], data_slice['ask_high'],
                          alpha=0.2, color='red', label='Ask Range (Low-High)')
    axes.plot(data_slice['timestamp'], data_slice['ask_close'], label='Ask Close', color='darkred', linewidth=1.5)

    axes.set_ylabel('Price', fontsize=12)
    axes.legend(loc='best')
    axes.set_title('Bid-Ask Prices and Ranges', fontsize=14)
    axes.grid(True, alpha=0.3)

    ax_volume = axes.twinx()
    x_indices = np.arange(len(data_slice))
    width = 0.4
    # Plot excess demand volume on the right y-axis
    excess_demand = data_slice['bid_volume'] - data_slice['ask_volume']
    ax_volume.bar(x_indices - width/2, data_slice['bid_volume'] - data_slice['ask_volume'], width=width,
                label='Excess Demand', color='green', alpha=0.6)
    ax_volume.set_ylabel('Excess Demand Volume', fontsize=12, color='green')
    ax_volume.tick_params(axis='y', labelcolor='green')
    ax_volume.legend(loc='upper right')

    plt.suptitle("Volume and Price Plot", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()
    return fig


def get_price(df, metric):
    assert metric in ["mid_price", "ask_bid_spread"]
    cols = df.shape[1]
    if cols == 5:
        bid_mid_price = df["bid"]
        ask_mid_price = df["ask"]
    else:
        bid_mid_price = (df["bid_open"] + df["bid_close"]) / 2
        ask_mid_price = (df["ask_open"] + df["ask_close"]) / 2
    
    if metric == "mid_price":
        mid_price = (bid_mid_price + ask_mid_price) / 2
        return mid_price
    elif metric == "ask_bid_spread":
        ask_bid_spread = (ask_mid_price - bid_mid_price) / 2
        return ask_bid_spread
    

def get_excess_demand(df):
    # Our independent variable is the excess demand. The Walrassian model predicts the change in price is due to
    # the mismatch in supply and demand. excess demand leads to increased prices.
    excess_demand = df["ask_volume"] - df["bid_volume"]
    excess_demand = excess_demand*1000
    return excess_demand

def regress_alpha_from_excess_volume(data, pred_window=25, plot=True, test_data=None):
    """
    Regress alpha from excess volume using different regression methods.

    Args:
        data: DataFrame with bid-ask data OR list of DataFrames (each representing a different day)
        use_open_close: Whether to use open/close prices (True) or high/low prices (False)
        pred_window: Window size for rolling average
        plot: Whether to plot results
        test_data: Optional test DataFrame or list of DataFrames for evaluation

    Returns:
        If test_data is None:
            coefficient: The alpha coefficient
            p_value: P-value (only available for L2 regression, None otherwise)
        If test_data is provided:
            coefficient: The alpha coefficient
            train_r2: R² score on training data
            test_r2: R² score on test data
            test_p_value: P-value on test data (only for L2 regression)
    """
    # Handle both single DataFrame and list of DataFrames
    if not isinstance(data, list):
        data = [data]

    # Process each DataFrame separately to avoid computing changes across days
    all_excess_demand = []
    all_mid_price_change = []

    for df in data:
        # We want to predict mid_price change from the excess demand: mid_{t+1} = mid_{t} + m (excess_volume)
        # If we want to smooth out thigns and compute this over a window
        mid_price = get_price(df, metric="mid_price")    
        mid_price_change = -1 * mid_price.diff(periods=-1)
        mid_price_change = mid_price_change.iloc[0:-1]      # the last value is NaN due to above
        mid_price_change = mid_price_change.rolling(window=pred_window).sum().shift(-pred_window+1)
        mid_price_change = mid_price_change.iloc[0:-pred_window]

        # Get excess demand
        excess_demand = get_excess_demand(df)
        excess_demand = excess_demand.iloc[0:-1]            # the last element cannot be used to predict
        excess_demand = excess_demand.rolling(window=pred_window).sum().shift(-pred_window+1)
        excess_demand = excess_demand.iloc[0:-pred_window]

        # Add to combined lists
        all_excess_demand.append(excess_demand)
        all_mid_price_change.append(mid_price_change)

    # Concatenate all data
    excess_demand = pd.concat(all_excess_demand, ignore_index=True)
    mid_price_change = pd.concat(all_mid_price_change, ignore_index=True)

    # Classical L2 (OLS) regression
    m = LinearRegression()
    m.fit(excess_demand.values.reshape(-1,1), mid_price_change.values)
    predicted_price_change = m.predict(excess_demand.values.reshape(-1,1))
    train_r2 = 1 - np.sum((mid_price_change.values - predicted_price_change)**2) / np.sum((mid_price_change.values - np.mean(mid_price_change.values))**2)
    
    # If test data is provided, evaluate on test set
    if test_data is not None:
        # Process test data using the same logic as training data
        if not isinstance(test_data, list):
            test_data = [test_data]

        all_test_excess_demand = []
        all_test_mid_price_change = []

        for df in test_data:
            mid_price_test = get_price(df, metric="mid_price")      
            mid_price_change_test = -1 * mid_price_test.diff(periods=-1)
            mid_price_change_test = mid_price_change_test.iloc[0:-1]      # the last value is NaN due to above
            mid_price_change_test = mid_price_change_test.rolling(window=pred_window).sum().shift(-pred_window+1)
            mid_price_change_test = mid_price_change_test.iloc[0:-pred_window]

            # Get excess demand
            excess_demand_test = get_excess_demand(df)
            excess_demand_test = excess_demand_test.iloc[0:-1]            # the last element cannot be used to predict
            excess_demand_test = excess_demand_test.rolling(window=pred_window).sum().shift(-pred_window+1)
            excess_demand_test = excess_demand_test.iloc[0:-pred_window]

            # Add to combined lists
            all_test_excess_demand.append(excess_demand_test)
            all_test_mid_price_change.append(mid_price_change_test)

        test_excess_demand = pd.concat(all_test_excess_demand, ignore_index=True)
        test_mid_price_change = pd.concat(all_test_mid_price_change, ignore_index=True)

        # Evaluate on test data
        test_predictions = m.predict(test_excess_demand.values.reshape(-1,1))
        test_r2 = 1 - np.sum((test_mid_price_change.values - test_predictions)**2) / np.sum((test_mid_price_change.values - np.mean(test_mid_price_change.values))**2)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot training data
            axes[0].scatter(excess_demand, mid_price_change, alpha=0.4, s=7)
            axes[0].plot(excess_demand, predicted_price_change, color="orange", linewidth=3)
            # Format coefficient in scientific notation with superscript
            coef_mantissa = m.coef_[0] / (10 ** np.floor(np.log10(np.abs(m.coef_[0]))))
            coef_exponent = int(np.floor(np.log10(np.abs(m.coef_[0]))))
            axes[0].set_title(r"\textbf{Training Data}" + f"($\\alpha={coef_mantissa:.2f} e^{{{coef_exponent}}}$, $R^2={train_r2:.4f}$)")
            axes[0].set_xlabel(r"\textbf{Excess Demand}")
            axes[0].set_ylabel(r"\textbf{Change in Price}")

            # Plot test data
            axes[1].scatter(test_excess_demand, test_mid_price_change, alpha=0.4, s=7, color='green')
            axes[1].plot(test_excess_demand, test_predictions, color="orange", linewidth=3)
            axes[1].set_title(f"$\\alpha$" + r"\textbf{ Regression - Test Data}" + f"($R^2={test_r2:.4f}$)")
            axes[1].set_xlabel(r"\textbf{Excess Demand}")
            axes[1].set_ylabel(r"\textbf{Change in Price}")

            plt.tight_layout()
            plt.show()

        return m.coef_[0], train_r2, test_r2

    # Original behavior when no test data is provided
    if plot:
        plt.figure()
        plt.scatter(excess_demand, mid_price_change, alpha=0.4, s=7)
        plt.plot(excess_demand, predicted_price_change, color="red")

        # Create title based on whether p-value is available
        # Format coefficient in scientific notation with superscript
        coef_mantissa = m.coef_[0] / (10 ** np.floor(np.log10(np.abs(m.coef_[0]))))
        coef_exponent = int(np.floor(np.log10(np.abs(m.coef_[0]))))
        title = f"$\\alpha$ Prediction - Training Data ($\\alpha={coef_mantissa:.2f} \\times 10^{{{coef_exponent}}}$, $R^2={train_r2:.4f}$)"
       
        plt.title(title)
        plt.xlabel("Excess Demand")
        plt.ylabel("Change in Price")
        plt.show()

    return m.coef_[0], train_r2


def regress_beta_from_excess_volume(data, pred_window=1, plot=True, test_data=None):
    """
    Regress beta from excess volume using different regression methods.

    Args:
        data: DataFrame with bid-ask data OR list of DataFrames (each representing a different day)
        pred_window: Window size for rolling average
        plot: Whether to plot results
        test_data: Optional test DataFrame or list of DataFrames for evaluation

    Returns:
        If test_data is None:
            coefficient: The beta coefficient
            train_r2: R² score on training data
        If test_data is provided:
            coefficient: The beta coefficient
            train_r2: R² score on training data
            test_r2: R² score on test data
    """
    # Handle both single DataFrame and list of DataFrames
    if not isinstance(data, list):
        data = [data]

    # Process each DataFrame separately to avoid computing changes across days
    all_excess_demand = []
    all_ask_bid_spread = []

    for df in data:
        # compute the ask-bid spreak. There two ways to do this.
        # (0) We can compute the avg of the bid open and close (same for ask) and then take the spread of these.
        # (1) We can compute the avg of the bid high and low (same for ask) and then take the spread of these.
        ask_bid_spread = get_price(df, metric="ask_bid_spread")    
        ask_bid_spread = ask_bid_spread.iloc[1:]            # can't predict the first item
        ask_bid_spread = ask_bid_spread.rolling(window=pred_window).sum().shift(-pred_window+1)
        ask_bid_spread = ask_bid_spread.iloc[0:-pred_window]

        # We want to predict ask-bid spread from the excess demand: spread = beta*(excess_volume)
        # Our independent variable is the excess demand.
        excess_demand = np.abs(get_excess_demand(df))
        excess_demand = excess_demand.iloc[0:-1]            # the element can be used to predict
        excess_demand = excess_demand.rolling(window=pred_window).sum().shift(-pred_window+1)
        excess_demand = excess_demand.iloc[0:-pred_window]

        # Add to combined lists
        all_excess_demand.append(excess_demand)
        all_ask_bid_spread.append(ask_bid_spread)

    # Concatenate all data
    excess_demand = pd.concat(all_excess_demand, ignore_index=True)
    ask_bid_spread = pd.concat(all_ask_bid_spread, ignore_index=True)

    # Classical L2 (OLS) regression
    m = LinearRegression()
    m.fit(excess_demand.values.reshape(-1,1), ask_bid_spread.values)
    predicted_spread = m.predict(excess_demand.values.reshape(-1,1))
    train_r2 = 1 - np.sum((ask_bid_spread.values - predicted_spread)**2) / np.sum((ask_bid_spread.values - np.mean(ask_bid_spread.values))**2)

    # If test data is provided, evaluate on test set
    if test_data is not None:
        # Process test data using the same logic as training data
        if not isinstance(test_data, list):
            test_data = [test_data]

        all_test_excess_demand = []
        all_test_ask_bid_spread = []

        for df in test_data:
            ask_bid_spread_test = get_price(df, metric="ask_bid_spread")    
            ask_bid_spread_test = ask_bid_spread_test.iloc[1:]
            ask_bid_spread_test = ask_bid_spread_test.rolling(window=pred_window).sum().shift(-pred_window+1)
            ask_bid_spread_test = ask_bid_spread_test.iloc[0:-pred_window]

            excess_demand_test = np.abs(get_excess_demand(df))
            excess_demand_test = excess_demand_test.iloc[0:-1]
            excess_demand_test = excess_demand_test.rolling(window=pred_window).sum().shift(-pred_window+1)
            excess_demand_test = excess_demand_test.iloc[0:-pred_window]

            all_test_excess_demand.append(excess_demand_test)
            all_test_ask_bid_spread.append(ask_bid_spread_test)

        test_excess_demand = pd.concat(all_test_excess_demand, ignore_index=True)
        test_ask_bid_spread = pd.concat(all_test_ask_bid_spread, ignore_index=True)

        # Evaluate on test data
        test_predictions = m.predict(test_excess_demand.values.reshape(-1,1))
        test_r2 = 1 - np.sum((test_ask_bid_spread.values - test_predictions)**2) / np.sum((test_ask_bid_spread.values - np.mean(test_ask_bid_spread.values))**2)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Plot training data
            axes[0].scatter(excess_demand, ask_bid_spread, alpha=0.4, s=7)
            axes[0].plot(excess_demand, predicted_spread, color="orange", linewidth=3)
            # Format coefficient in scientific notation with superscript
            coef_mantissa = m.coef_[0] / (10 ** np.floor(np.log10(np.abs(m.coef_[0]))))
            coef_exponent = int(np.floor(np.log10(np.abs(m.coef_[0]))))
            axes[0].set_title(r"\textbf{Training Data}" + f"($\\beta={coef_mantissa:.2f} e^{{{coef_exponent}}}$, $R^2={train_r2:.4f}$)")
            axes[0].set_xlabel(r"\textbf{Absolute Excess Demand}")
            axes[0].set_ylabel(r"\textbf{Ask-Bid Spread}")
            
            # Plot test data
            axes[1].scatter(test_excess_demand, test_ask_bid_spread, alpha=0.4, s=7)
            axes[1].plot(test_excess_demand, test_predictions, color="orange", linewidth=3)
            axes[1].set_title(f"$\\beta$" + r" \textbf{Regression - Test Data}" + f"($R^2={test_r2:.4f}$)")
            axes[1].set_xlabel(r"\textbf{Absolute Excess Demand}")
            axes[1].set_ylabel(r"\textbf{Ask-Bid Spread}")

            plt.tight_layout()
            plt.show()

        return m.coef_[0], train_r2, test_r2

    # Original behavior when no test data is provided
    if plot:
        plt.figure()
        plt.scatter(excess_demand, ask_bid_spread, alpha=0.4, s=7)
        plt.plot(excess_demand, predicted_spread, color="red")

        # Create title based on whether p-value is available
        # Format coefficient in scientific notation with superscript
        coef_mantissa = m.coef_[0] / (10 ** np.floor(np.log10(np.abs(m.coef_[0]))))
        coef_exponent = int(np.floor(np.log10(np.abs(m.coef_[0]))))
        title = f"$\\beta Prediction - Training Data ($\\beta={coef_mantissa:.2f} e^{{{coef_exponent}}}$, $R^2={train_r2:.4f}$)"

        plt.title(title)
        plt.xlabel("Excess Demand")
        plt.ylabel("Ask-Bid Spread")
        plt.show()

    return m.coef_[0], train_r2


def get_histogram(tick_files, alpha_window, beta_window):
    """
    Compute alpha and beta for multiple datasets and create a scatter plot.

    Args:
        tick_files: List of tick data file paths
        test_files: Optional list of test data file paths

    Returns:
        alphas, betas: Arrays of computed alpha and beta values
    """
    dataset = convert_tick_data_to_pd(tick_files)
    alphas, betas = np.zeros(len(tick_files)), np.zeros(len(tick_files))
    for i, data in enumerate(dataset):
        alpha, alpha_train_r2 = regress_alpha_from_excess_volume(
            data,
            pred_window=alpha_window,
            plot=False
        )
        beta, beta_train_r2 = regress_beta_from_excess_volume(
            data,
            pred_window=beta_window,
            plot=False,
        )
        alphas[i] = alpha
        betas[i] = beta

    # Make a histogram of alpha values
    plt.figure(figsize=(8, 6))
    plt.hist(alphas, bins=20, alpha=0.7, edgecolor='black', linewidth=1.2)
    plt.xlabel(r'$\alpha$ (Perm. Impact)')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of $\alpha$ Values')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    # Make a histogram of beta values
    plt.figure(figsize=(8, 6))
    plt.hist(betas, bins=15, alpha=0.7, edgecolor='black', linewidth=1.2)
    plt.xlabel(r'$\beta$ (Temp. Impact)')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of $\beta$ Values')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    return alphas, betas


def get_supply_vector():
    # Let's get the supply for a single hour of a single day. We will play 
    # strategic against this exogenous vector.
    data_folder = "data_tick/train_tuesday_10am_11am_12pm"
    data_dir = os.path.join(PANDAS_FOLDER_NAME, data_folder)
    tick_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    tick_files = [os.path.join(data_dir, tick_file) for tick_file in tick_files]
    data = convert_tick_data_to_pd(tick_files)
    data = data[0]
    excess_demand = get_excess_demand(data)
    
    total_volume = data["ask_volume"] + data["bid_volume"]
    total_volume = total_volume*1000
    return excess_demand, total_volume


def get_and_plot_alpha_beta():
    PLOT_HIST = False
    PLOT_REG = True
    
    # Automatically get all CSV files in the data folder
    data_folder = "data_tick/train_tuesday_10am_11am_12pm"
    data_dir = os.path.join(PANDAS_FOLDER_NAME, data_folder)
    tick_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')])
    tick_files = [os.path.join(data_dir, tick_file) for tick_file in tick_files]
    data = convert_tick_data_to_pd(tick_files)
    print(f"Found {len(tick_files)} CSV files in {data_dir} for Training")

    test_folder = "data_tick/test_tuesday_10am_11am_12pm"
    test_data_dir = os.path.join(PANDAS_FOLDER_NAME, test_folder)
    test_files = sorted([f for f in os.listdir(test_data_dir) if f.endswith('.csv')])
    test_files = [os.path.join(test_data_dir, test_file) for test_file in test_files]
    test_data = convert_tick_data_to_pd(test_files)
    print(f"Found {len(test_files)} CSV files in {test_data_dir} for Testing")

    alpha_window, beta_window = 100, 40
    
    if PLOT_HIST:
        get_histogram(tick_files, alpha_window, beta_window)

    alpha, alpha_train_r2, alpha_test_r2 = regress_alpha_from_excess_volume(
        data,
        pred_window=alpha_window,
        plot=PLOT_REG,
        test_data=test_data
    )
    print(f"Alpha with window {alpha_window}- Coefficient: {alpha:.2e}, Train R²: {alpha_train_r2:.4f}, Test R²: {alpha_test_r2:.2e}")
    beta, beta_train_r2, beta_test_r2 = regress_beta_from_excess_volume(
        data,
        pred_window=beta_window,
        plot=PLOT_REG,
        test_data=test_data
    )
    print(f"Beta with window {beta_window} - Coefficient: {beta:.2e}, Train R²: {beta_train_r2:.4f}, Test R²: {beta_test_r2:.2e}")


if __name__ == "__main__":
    get_and_plot_alpha_beta()
