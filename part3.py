import pandas as pd
import re

def process_text(text):
    if not isinstance(text, str):  # Ensure input is a string
        return ""

    text = text.lower()  # Convert to lowercase

    # Replace URLs
    text = re.sub(r"([^ \n]*\.[^ \n.]{2,})", "<URL>", text, flags=re.MULTILINE)

    # Replace Emails
    text = re.sub(r"([^ \n]*@[^ \n.]{1,})", "<Mail>", text, flags=re.MULTILINE)

    # Replace Dates (Month & Weekday names)
    text = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
                  r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "<Date>", text, flags=re.IGNORECASE)

    # Replace Numbers
    text = re.sub(r"\d+", "<Num>", text)

    # Remove multiple spaces, tabs, and newlines
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Load CSV file
Mydata = pd.read_csv("news_sample.csv")

# Apply function and replace the original 'content' column
Mydata["content"] = Mydata["content"].apply(process_text)

# Save cleaned data back to CSV (if needed)
Mydata.to_csv("cleaned_news_sample.csv", index=False)

# Display first few rows
#print(Mydata.head())


print(Mydata["content"])

