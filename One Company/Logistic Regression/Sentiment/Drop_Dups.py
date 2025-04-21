import pandas as pd

# Load the CSV
df = pd.read_csv("JNJNoSent.csv")

# Drop duplicate rows
df = df.drop_duplicates()

# Save it back
df.to_csv("JNJNoSent_NoDups.csv", index=False)
