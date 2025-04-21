import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from Inputs_No_Sentiment import * #Get All Functions from Helper File
from Outputs_No_Sentiment_All import * #Get All Functions from Helper File

def single_window(START_DATE, END_DATE, OUTPUT_CSV):
    
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

    all_data = []
    future_closing_prices = get_future_closing_prices(companies, START_DATE, END_DATE)
    #print("Hello: {}".format(future_closing_prices))
    i = 0

    for company in companies:
        ticker = company["ticker"]
        name = company["name"]
        print(f"Processing {name} ({ticker})...") #For when Code Runs
        try:
            # --- Download historical stock data ---
            """We add a buffer of 30 days before START_DATE to ensure we have enough
            historical data for rolling-window calculations (e.g., ADX, RSI, etc.).
            Many indicators require multiple prior periods to compute their first valid values.
            his buffer ensures accurate indicator values within our target date range. """
            
            buffer_days = 30
            start_for_calc = pd.to_datetime(START_DATE) - pd.Timedelta(days=buffer_days)

            df = yf.download(ticker, start=start_for_calc.strftime('%Y-%m-%d'), end=END_DATE, auto_adjust=True)

            if df.empty:
                print(f"No data for {ticker}") #When Code Runs, it will Tell Us if Data wasn't Retreived
                continue

            #Don't Worry about this. Just formats the dataframe correctly.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Compute the technical indicators
            indicators_df = compute_indicators(df)
            indicators_df['Company'] = name
            indicators_df['Ticker'] = ticker
            indicators_df['StartDate'] = START_DATE
            indicators_df['EndDate'] = END_DATE
            

            
            """ Reorder columns to place identifying metadata first (Ticker, Company, StartDate, EndDate),
            followed by the computed technical indicators. This improves readability and consistency
            in the final CSV output. """


            ordered_cols = ['Ticker', 'Company', 'StartDate', 'EndDate'] + [col for col in indicators_df.columns if col not in ['Ticker', 'Company', 'StartDate', 'EndDate']]
            indicators_df = indicators_df[ordered_cols]
            print(future_closing_prices[i])
            indicators_df["Output"] = future_closing_prices[i].item()

            i+=1

            all_data.append(indicators_df)

            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        if os.path.exists(OUTPUT_CSV):
            existing_df = pd.read_csv(OUTPUT_CSV)
            combined_df = pd.concat([existing_df, combined_df], ignore_index=True)

        combined_df.to_csv(OUTPUT_CSV, index=False)
        print(f"CSV file '{OUTPUT_CSV}' updated.")

    else:
        print("No data to write to CSV.")
