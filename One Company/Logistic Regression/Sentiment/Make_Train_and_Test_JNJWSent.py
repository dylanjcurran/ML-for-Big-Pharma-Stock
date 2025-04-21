from One_Window_JNJ_Reg import *
from datetime import datetime, timedelta
from Reddit_Sent_Dup import *
import pandas as pd
import os


#--- Global Variables ---

recent = '2024-08-01'  # Proves Enough Buffer to Get 6 Months Ahead
num_windows = 200
window_size_in_days = 30


#--- Helper Functions ---

def N_days_ago(N, date):
    """
    Subtract N days from a datetime object and return the result.
    """
    if type(date) == str:
        return datetime.strptime(date, "%Y-%m-%d") - timedelta(days=int(N))
    return date - timedelta(days=int(N))


def format_ymd(date_obj):
    
    return date_obj.strftime("%Y-%m-%d")

def getDates():
    dates = []
    day = datetime.strptime(recent, "%Y-%m-%d")  # convert only once at the start
    for i in range(num_windows):
        first_date = (N_days_ago(window_size_in_days, day))
        if type(day) == str:
            pass
        else:
            day = format_ymd(day)
        
        dates.append((format_ymd(first_date), day))
        day = format_ymd(N_days_ago(window_size_in_days / 2, day))  # Half Overlap

    dates.reverse()
    
    return dates


#--- Main Script ---

#Get Dates
dates = getDates()

#Stuff

csv_file = "JNJ.csv"
i = 1
num_windows = len(dates)

for start, end in dates:
    # Step 0: Read file if it exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame()

    # Step 1: Skip if this window is already fully processed
    if not df.empty and ((df["StartDate"] == start) & (df["EndDate"] == end)).any():
        print(f"Skipping duplicate window: {start} to {end}")
        continue

    # Step 2: Append technical data
    single_window(start, end, csv_file)

    # Step 3: Re-load file and target the most recent row
    df = pd.read_csv(csv_file)

    # Step 4: Get sentiment data for this window
    sent_features = get_reddit_sentiment_custom_window(
        start_date=start,
        end_date=end,
        keywords=["jnj", "johnson", "johnson and johnson"],
        subreddits=["stocks", "investing", "wallstreetbets"],
        limit=500
    )
    sent_features["StartDate"] = start
    sent_features["EndDate"] = end

    # Step 5: Write sentiment data to the *last* row
    for key, val in sent_features.items():
        df.at[df.index[-1], key] = val

    # Step 6: Move 'Output' to the end
    cols = list(df.columns)
    if "Output" in cols:
        cols.remove("Output")
        cols.append("Output")
    df = df[cols]

    # Step 7: Save the updated file
    df.to_csv(csv_file, index=False)

    print(f"{i} out of {num_windows} done")
    i += 1

# Step 8: Final cleanup (optional double safety)
df = pd.read_csv(csv_file)
df = df.drop_duplicates(subset=["StartDate", "EndDate"], keep="last")
df.to_csv(csv_file, index=False)
