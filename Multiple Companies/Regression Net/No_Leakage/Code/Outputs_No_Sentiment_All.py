import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

companies = [
    {"name": "Johnson & Johnson", "ticker": "JNJ"},]


def determine_look_ahead_days(start_date_str, end_date_str):
    """
    This function determines how many days after the window we should pull
    stock data for the outputs to our training data. The longer the window,
    the longer ahead the date should be to reflect the different kinds of
    trading (i.e day trading, long-term trading, etc).
    """
    
    fmt = "%Y-%m-%d"
    start_dt = datetime.strptime(start_date_str, fmt)
    end_dt = datetime.strptime(end_date_str, fmt)

    window_length = (end_dt - start_dt).days

    if window_length < 7: #Up to a Week
        return 7         #Do a Week
    elif window_length < 30: #Up to a Month
        return 14        # Do 2 Weeks
    elif window_length < 180: #Up to 6 Months
        return 30        # Do a Month
    elif window_length < 365: #Up to a Year
        return 60        # Do 2 Months
    elif window_length < 730: #Up to 2 Years
        return 90        # Do 3 Months
    else: #Anything Longer
        return 180       # Do 6 Months


def get_historical_data(companies, start_date, end_date):
    """
    Pull historical data for each company's ticker from
    start_date to end_date using yfinance.
    """
    data_dict = {}
    for comp in companies:
        ticker = comp["ticker"]
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        data_dict[ticker] = df
        #print(df) --> Just Making Sure Code Works
        #break --> Didn't Want Loop to Run Over Again when Testing
    return data_dict


def get_future_closing_prices(companies, start_date, end_date):
    """
    Returns the future price difference (future_close - current_close)
    for each company using a dynamically determined lookahead window.

    Assumes this is called *per window* and does not cache data across windows
    (to avoid leakage in overlapping scenarios).
    """
    look_ahead_days = determine_look_ahead_days(start_date, end_date)

    fmt = "%Y-%m-%d"
    start_dt = datetime.strptime(start_date, fmt)
    end_dt = datetime.strptime(end_date, fmt)
    
    future_start = (end_dt + timedelta(days=1)).strftime(fmt)
    future_end = (end_dt + timedelta(days=look_ahead_days)).strftime(fmt)

    closing_price_list = []

    for comp in companies:
        ticker = comp["ticker"]
        
        # Pull past and future data separately per company
        df_past = yf.download(ticker, start=start_date, end=end_date, progress=False)
        df_future = yf.download(ticker, start=future_start, end=future_end, progress=False)

        if not df_past.empty and not df_future.empty:
            current_close = df_past['Close'].iloc[-1]
            future_close = df_future['Close'].iloc[-1]
            price_diff = future_close - current_close
        else:
            price_diff = 0.0  # Could also use np.nan or 0.0 if desired

        closing_price_list.append(price_diff)

    return closing_price_list
