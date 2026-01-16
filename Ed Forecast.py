# deployment code (updated with new features)
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


def create_post_rest_spike_feature(df, date_col='ds', year_list=None):
    """
    Marks the first working day after 3+ consecutive rest days (weekends + holidays) as 1.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if year_list is None:
        year_list = df[date_col].dt.year.unique().tolist()
    
    sg_holidays = make_holidays_df(year_list=year_list, country='SG')
    holiday_dates = set(pd.to_datetime(sg_holidays['ds']).dt.date)
    
    min_date = df[date_col].min() - pd.Timedelta(days=10)
    max_date = df[date_col].max() + pd.Timedelta(days=10)
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    helper_df = pd.DataFrame({'date': all_dates})
    helper_df['is_weekend'] = helper_df['date'].dt.dayofweek.isin([5, 6])
    helper_df['is_holiday'] = helper_df['date'].dt.date.isin(holiday_dates)
    helper_df['is_rest_day'] = helper_df['is_weekend'] | helper_df['is_holiday']
    helper_df['rest_group'] = (~helper_df['is_rest_day']).cumsum()
    
    spike_dates = []
    for group_id, group_df in helper_df[helper_df['is_rest_day']].groupby('rest_group'):
        if len(group_df) >= 3:
            last_rest_day = group_df['date'].max()
            next_day = last_rest_day + pd.Timedelta(days=1)
            next_day_info = helper_df[helper_df['date'] == next_day]
            if len(next_day_info) > 0 and not next_day_info['is_rest_day'].values[0]:
                spike_dates.append(next_day)
    
    df['post_rest_spike'] = df[date_col].isin(set(spike_dates)).astype(int)
    return df


def create_post_holiday_features(df, date_col='ds', year_list=None):
    """
    Creates binary columns marking the next working day after specific holidays:
    - after_cny: Next working day after Chinese New Year
    - after_christmas: Next working day after Christmas Day
    - after_ny: Next working day after New Year's Day
    - after_ramadan: Next working day after Eid al-Fitr
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    if year_list is None:
        year_list = df[date_col].dt.year.unique().tolist()
    
    # Get Singapore holidays
    sg_holidays = make_holidays_df(year_list=year_list, country='SG')
    sg_holidays['ds'] = pd.to_datetime(sg_holidays['ds'])
    
    # Create set of all holiday dates for checking working days
    all_holiday_dates = set(sg_holidays['ds'].dt.date)
    
    def get_next_working_day(holiday_date, all_holiday_dates):
        """Find the next working day after a holiday (not weekend, not holiday)"""
        next_day = holiday_date + pd.Timedelta(days=1)
        # Keep advancing until we find a working day
        while True:
            is_weekend = next_day.dayofweek in [5, 6]  # Saturday=5, Sunday=6
            is_holiday = next_day.date() in all_holiday_dates
            if not is_weekend and not is_holiday:
                return next_day
            next_day += pd.Timedelta(days=1)
    
    # Define holiday patterns to match
    holiday_patterns = {
        'after_cny': ['Chinese New Year', 'Chinese New Year (observed)'],
        'after_christmas': ['Christmas Day', 'Christmas Day (observed)'],
        'after_ny': ["New Year's Day", "New Year's Day (observed)"],
        'after_ramadan': ['Eid al-Fitr', 'Eid al-Fitr (observed)']
    }
    
    # Find post-holiday dates for each category
    post_holiday_dates = {key: set() for key in holiday_patterns.keys()}
    
    for feature_name, patterns in holiday_patterns.items():
        # Get all dates for this holiday type
        holiday_dates = sg_holidays[sg_holidays['holiday'].isin(patterns)]['ds'].tolist()
        
        # For holidays that span multiple days (like CNY), find the last day
        # Group consecutive dates
        if holiday_dates:
            holiday_dates_sorted = sorted(holiday_dates)
            
            # Group consecutive holiday dates
            groups = []
            current_group = [holiday_dates_sorted[0]]
            
            for i in range(1, len(holiday_dates_sorted)):
                # Check if this date is consecutive to the previous one
                if (holiday_dates_sorted[i] - holiday_dates_sorted[i-1]).days <= 1:
                    current_group.append(holiday_dates_sorted[i])
                else:
                    groups.append(current_group)
                    current_group = [holiday_dates_sorted[i]]
            groups.append(current_group)
            
            # For each group, find the next working day after the last date
            for group in groups:
                last_holiday_date = max(group)
                next_working = get_next_working_day(last_holiday_date, all_holiday_dates)
                post_holiday_dates[feature_name].add(next_working)
    
    # Create binary columns
    for feature_name, dates in post_holiday_dates.items():
        df[feature_name] = df[date_col].isin(dates).astype(int)
    
    return df


def add_all_features(df, year_list=None):
    """Add all features: F1 events, post-rest spike, and post-holiday features."""
    if year_list is None:
        year_list = df['ds'].dt.year.unique().tolist()
    
    # Add F1 events
    df = add_f1_events(df)
    
    # Add post-rest spike feature
    df = create_post_rest_spike_feature(df, date_col='ds', year_list=year_list)
    
    # Add post-holiday features
    df = create_post_holiday_features(df, date_col='ds', year_list=year_list)
    
    return df


def train_model(df, years=[2022, 2023, 2024, 2025]):
    """Train Prophet model on specified years with all regressors."""
    df_train = df[df["ds"].dt.year.isin(years)].dropna(subset=["last_30_avg"])
    
    holidays = make_holidays_df(year_list=years, country="SG")
    holidays["lower_window"] = 0
    holidays["upper_window"] = 0
    
    model = Prophet(holidays=holidays, seasonality_mode="multiplicative")
    model.add_regressor("covid_cases")
    model.add_regressor("f1_event")
    model.add_regressor("last_30_avg")
    model.add_regressor("post_rest_spike")
    model.add_regressor("after_cny")
    model.add_regressor("after_christmas")
    model.add_regressor("after_ny")
    model.add_regressor("after_ramadan")
    model.fit(df_train)
    
    return model


def prepare_future_features(df_future, df_historical, prediction_date, year_list):
    """Prepare all feature values for future dates."""
    prediction_date = pd.to_datetime(prediction_date)
    
    # COVID cases (assume 0 for future)
    df_future["covid_cases"] = 0
    
    # Last 30 day average from historical data
    df_future["last_30_avg"] = df_historical[df_historical["ds"] <= prediction_date]["y"].tail(30).mean()
    
    # F1 events
    f1_dates = get_f1_dates(year_list)
    df_future["f1_event"] = df_future["ds"].isin(f1_dates).astype(int)
    
    # Apply F1 lag effects (4 days before, 2 days after)
    for idx, row in df_future.iterrows():
        if row["f1_event"] == 0:
            for f1_date in f1_dates:
                diff = (row["ds"] - f1_date).days
                if -4 <= diff <= 2:
                    df_future.loc[idx, "f1_event"] = 1
                    break
    
    # Post-rest spike feature
    df_future = create_post_rest_spike_feature(df_future, date_col='ds', year_list=year_list)
    
    # Post-holiday features
    df_future = create_post_holiday_features(df_future, date_col='ds', year_list=year_list)
    
    return df_future


def predict(model, df, prediction_date):
    """Predict 1, 2, 3 days ahead from prediction_date."""
    prediction_date = pd.to_datetime(prediction_date)
    future_dates = [prediction_date + timedelta(days=i) for i in [1, 2, 3]]
    
    # Build future dataframe
    df_future = pd.DataFrame({"ds": future_dates})
    
    # Determine year list for feature generation
    year_list = list(set([d.year for d in future_dates] + [2022, 2023, 2024, 2025, 2026]))
    
    # Prepare all features
    df_future = prepare_future_features(df_future, df, prediction_date, year_list)
    
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
    
    print("Adding features (F1, post-rest spike, post-holiday)...")
    year_list = [2022, 2023, 2024, 2025, 2026]
    df = add_all_features(df, year_list=year_list)
    
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
    input_file = r"C:\Users\thach\VSCodeProjects\cgh-project-updated\data\edarrivals_20182024.csv"
    prediction_date = "2024-12-31"  # or None to use latest date
    output_file = "predictions.csv"  # or None to skip saving
    
    run_forecast(input_file, prediction_date, output_file)