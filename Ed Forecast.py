# To use, replace file paths / names at IF statement at the very bottom 

import sys
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import fastf1

warnings.filterwarnings("ignore")

# Fallback F1 Singapore GP dates (start, end) - used if fastf1 fails
F1_DATES_FALLBACK = {
    2022: ("2022-09-30", "2022-10-02"),
    2023: ("2023-09-15", "2023-09-17"),
    2024: ("2024-09-20", "2024-09-22"),
    2025: ("2025-10-03", "2025-10-05"),
    2026: ("2026-10-02", "2026-10-04"),
}


def get_f1_dates(years):
    f1_dates = []
    
    for year in years:
        if year in [2020, 2021]:  # F1 cancelled these years
            continue
        
        try:
            schedule = fastf1.get_event_schedule(year)
            sg_race = schedule[schedule["Country"] == "Singapore"]
            if not sg_race.empty:
                start = pd.to_datetime(sg_race["Session1Date"].iloc[0]).tz_localize(None).normalize()
                end = pd.to_datetime(sg_race["EventDate"].iloc[0]).tz_localize(None).normalize()
                f1_dates.extend(pd.date_range(start, end))
                continue
        except:
            pass
        
        # Fallback to hardcoded dates
        if year in F1_DATES_FALLBACK:
            start, end = F1_DATES_FALLBACK[year]
            f1_dates.extend(pd.date_range(start, end))
    
    return f1_dates


def load_data(filepath):
    """Load CSV or Excel file."""
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, low_memory=False)
    return pd.read_excel(filepath)


def preprocess(df):
    """Convert raw data to daily counts with features."""
    df = df.copy()
    
    # Remove cancellations
    if "A&E Discharge Type Description" in df.columns:
        df = df[df["A&E Discharge Type Description"] != "Cancellation"]
    
    # Parse dates and count daily arrivals
    df["A&E Admit Date"] = pd.to_datetime(df["A&E Admit Date"].astype(str).str.strip(), format="mixed", dayfirst=True)
    daily = df.value_counts("A&E Admit Date").reset_index()
    daily.columns = ["ds", "y"]
    daily = daily.sort_values("ds").reset_index(drop=True)
    
    # COVID cases per day
    if "A&E Diagnosis Code" in df.columns:
        covid = df[df["A&E Diagnosis Code"].isin(["B342", "B972"])].groupby("A&E Admit Date").size().reset_index(name="covid_cases")
        covid.columns = ["ds", "covid_cases"]
        daily = daily.merge(covid, on="ds", how="left")
    daily["covid_cases"] = daily.get("covid_cases", 0).fillna(0).astype(int)
    
    # 30-day rolling average (lagged by 1 day)
    daily["last_30_avg"] = daily["y"].shift(1).rolling(30).mean()
    
    return daily


def add_f1_events(df):
    """Add F1 event indicator with lag effects."""
    years = list(range(2022, 2027))
    f1_dates = get_f1_dates(years)
    df_f1 = pd.DataFrame({"ds": f1_dates, "f1_event": 1})
    
    # Merge and apply lag (4 days before, 2 days after)
    df = df.merge(df_f1, on="ds", how="left")
    df["f1_event"] = df["f1_event"].fillna(0)
    temp = df["f1_event"].replace(0, np.nan)
    df["f1_event"] = temp.bfill(limit=4).ffill(limit=2).fillna(0).astype(int)
    
    return df


def train_model(df, years=[2022, 2023, 2024, 2025]):
    """Train Prophet model on specified years."""
    df_train = df[df["ds"].dt.year.isin(years)].dropna(subset=["last_30_avg"])
    
    holidays = make_holidays_df(year_list=years, country="SG")
    holidays["lower_window"] = -3
    holidays["upper_window"] = 0
    
    model = Prophet(holidays=holidays, seasonality_mode="multiplicative")
    model.add_regressor("covid_cases")
    model.add_regressor("f1_event")
    model.add_regressor("last_30_avg")
    model.fit(df_train)
    
    return model


def predict(model, df, prediction_date):
    """Predict 1, 2, 3 days ahead from prediction_date."""
    prediction_date = pd.to_datetime(prediction_date)
    future_dates = [prediction_date + timedelta(days=i) for i in [1, 2, 3]]
    
    # Build future dataframe
    df_future = pd.DataFrame({"ds": future_dates})
    df_future["covid_cases"] = 0
    df_future["last_30_avg"] = df[df["ds"] <= prediction_date]["y"].tail(30).mean()
    
    # Add F1 indicator
    years = list(set([d.year for d in future_dates] + [2022, 2023, 2024, 2025, 2026]))
    f1_dates = get_f1_dates(years)
    df_future["f1_event"] = df_future["ds"].isin(f1_dates).astype(int)
    
    # Apply F1 lag effects
    for idx, row in df_future.iterrows():
        if row["f1_event"] == 0:
            for f1_date in f1_dates:
                diff = (row["ds"] - f1_date).days
                if -4 <= diff <= 2:
                    df_future.loc[idx, "f1_event"] = 1
                    break
    
    # Predict
    forecast = model.predict(df_future)
    
    results = pd.DataFrame({
        "prediction_date": prediction_date,
        "target_date": forecast["ds"],
        "days_ahead": [1, 2, 3],
        "predicted_arrivals": forecast["yhat"].round(0).astype(int),
        "lower_bound": forecast["yhat_lower"].round(0).astype(int),
        "upper_bound": forecast["yhat_upper"].round(0).astype(int),
    })
    
    return results


def run_forecast(input_file, prediction_date=None, output_file=None):
    """Main function: load data, train, and predict."""
    print(f"Loading {input_file}...")
    df_raw = load_data(input_file)
    
    print("Preprocessing...")
    df = preprocess(df_raw)
    df = add_f1_events(df)
    
    print("Training model...")
    model = train_model(df)
    
    # Use latest date if not specified
    if prediction_date is None:
        prediction_date = df["ds"].max()
    
    print(f"Predicting from {pd.to_datetime(prediction_date).date()}...")
    results = predict(model, df, prediction_date)
    
    # Print results
    print("\n" + "=" * 50)
    print("PREDICTIONS")
    print("=" * 50)
    for _, row in results.iterrows():
        print(f"{row['target_date'].strftime('%Y-%m-%d')} ({row['days_ahead']}d ahead): "
              f"{row['predicted_arrivals']} (CI: {row['lower_bound']}-{row['upper_bound']})")
    print("=" * 50)
    
    # Save if requested
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nSaved to {output_file}")
    
    return results


if __name__ == "__main__":

    # Replace settings here
    # Need to update input file with raw data, make sure format is same as current excel files provided by CGH
    # Need to update the input file daily for best results
    input_file = r"C:\Users\thach\VSCodeProjects\cgh-project-updated\data\edarrivals_20182024.csv"
    prediction_date = "2024-12-31"  # or None to use latest date
    output_file = "predictions.csv"  # or None to skip saving
    

    run_forecast(input_file, prediction_date, output_file)


