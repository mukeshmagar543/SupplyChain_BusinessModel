import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data():
    url = "https://raw.githubusercontent.com/mukeshmagar543/SupplyChain_BusinessModel/refs/heads/main/research/SCM.csv"
    logging.info("Loading data from URL")
    df = pd.read_csv(url)
    return df