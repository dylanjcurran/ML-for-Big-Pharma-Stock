import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

companies = [
    {"name": "Johnson & Johnson", "ticker": "JNJ"},
    {"name": "Pfizer Inc.", "ticker": "PFE"},
    {"name": "Merck & Co., Inc.", "ticker": "MRK"},
    {"name": "Roche Group", "ticker": "RHHBY"},
    {"name": "AbbVie Inc.", "ticker": "ABBV"},
    {"name": "Novartis AG", "ticker": "NVS"},
    {"name": "Bayer", "ticker": "BAYRY"},
    {"name": "Eli Lilly and Company", "ticker": "LLY"},
    {"name": "AstraZeneca PLC", "ticker": "AZN"},
    {"name": "Novo Nordisk A/S", "ticker": "NVO"},
    {"name": "Bristol-Myers Squibb Company", "ticker": "BMY"},
    {"name": "Sanofi", "ticker": "SNY"},
    {"name": "Abbott Laboratories", "ticker": "ABT"},
    {"name": "GSK plc", "ticker": "GSK"},
    {"name": "Amgen Inc.", "ticker": "AMGN"},
    {"name": "Gilead Sciences, Inc.", "ticker": "GILD"},
    {"name": "Takeda Pharmaceutical Company Limited", "ticker": "TAK"},
    {"name": "Viatris Inc.", "ticker": "VTRS"},
    {"name": "Daiichi Sankyo", "ticker": "DSNKY"},
    {"name": "Biogen Inc.", "ticker": "BIIB"},
    {"name": "Astellas Pharma", "ticker": "ALPMY"},
    {"name": "Otsuka Holdings", "ticker": "OTSKY"},
    {"name": "Teva Pharmaceutical Industries Limited", "ticker": "TEVA"}
]


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
    1. Determine how many days to look ahead based on the length of the window.
    This uses the defined function above.
    2. Pull stock data from (end_date+1) to (end_date + look_ahead_days).
    3. For each company, take the last available Close price in that future range.
    For those curious, we do this because a fixed look-ahead day is not guaranteed
    to be a day in which the stock market is active.
    4. Return a list of closing prices in the same order as 'companies'.
    """
    look_ahead_days = determine_look_ahead_days(start_date, end_date)

    fmt = "%Y-%m-%d"
    end_dt = datetime.strptime(end_date, fmt)
    
    # Future window: from the next calendar day ...
    future_start = (end_dt + timedelta(days=1)).strftime(fmt)
    # ... to (end_date + look_ahead_days)
    future_end = (end_dt + timedelta(days=look_ahead_days)).strftime(fmt)

    future_data_dict = {}
    for comp in companies:
        ticker = comp["ticker"]
        df_future = yf.download(ticker, start=future_start, end=future_end, progress=False)
        future_data_dict[ticker] = df_future

    # Build a list of final closing prices
    closing_price_list = []
    for comp in companies:
        ticker = comp["ticker"]
        df_future = future_data_dict[ticker]

        if not df_future.empty:
            last_close = df_future['Close'].iloc[-1].item()
        else:
            # If no data, store None (or 0, or any placeholder)
            last_close = None

        closing_price_list.append(last_close)

    return closing_price_list
