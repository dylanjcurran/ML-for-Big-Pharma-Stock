from One_Window_JNJ_Reg import *
from datetime import datetime, timedelta
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

csv_file = "JNJNoSent.csv"
i = 1
num_windows = len(dates)

for (start, end) in dates:
    single_window(start, end, csv_file)
    print("{} out of {}".format(i, num_windows))
    i += 1
