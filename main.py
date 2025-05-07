import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

# --- Imports for new Random Forest Section ---
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import uuid # Not strictly needed from snippet but included if used later

# ------------------ Load Data ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("C:/Filse/smai_new-data/raw/cleaned_data/latest_code.csv")
    except FileNotFoundError:
        st.error("Failed to load 'latest_code.csv'. Please ensure the file path is correct.")
        st.info("Attempting to load a placeholder 'ufc-fighters-statistics.csv' for demonstration.")
        try:
            df = pd.read_csv("ufc-fighters-statistics.csv")
        except FileNotFoundError:
            st.warning("Placeholder dataset not found. Some features might not work as expected.")
            df = pd.DataFrame()

    if df.empty:
        return df

    required_cols_for_dropna = ["winner", "SigStr1_landed", "Ctrl1", "KD1", "Td1_landed"]
    if not all(col in df.columns for col in required_cols_for_dropna):
        st.warning(f"DataFrame is missing some required columns for initial dropna. App may not function correctly.")
        for col in required_cols_for_dropna:
            if col not in df.columns: df[col] = np.nan if col == "winner" else 0
        if "winner" not in df.columns: df["winner"] = "Unknown" # Critical for Win calculation

    df = df.dropna(subset=[col for col in required_cols_for_dropna if col in df.columns])
    
    if "Fighter1" not in df.columns: df["Fighter1"] = "UnknownFighter1"
    if "Fighter2" not in df.columns: df["Fighter2"] = "UnknownFighter2"

    if "winner" in df.columns:
        df["Win"] = (df['winner'].str.lower() == df['Fighter1'].str.lower()).astype(int)
    else:
        df["Win"] = 0

    return df

df_full_data = load_data()

if df_full_data.empty:
    st.error("Data could not be loaded. Dashboard cannot operate.")
    st.stop()

df = df_full_data.copy()


st.sidebar.header("⚙️ Data Filtering & Options")
st.subheader("🔍 Data Overview")
st.write(f"Loaded {len(df)} fight records.")

if 'Fight_Duration_Min' not in df.columns:
    df['Fight_Duration_Min'] = 15.0 # Default assumption

possible_date_columns = ['Event Date', 'event_date', 'Date', 'date', 'event date']
event_date_col = None
for col in possible_date_columns:
    if col in df.columns:
        event_date_col = col
        break

is_synthetic_dates = False
if event_date_col and event_date_col != 'Event Date':
    df.rename(columns={event_date_col: 'Event Date'}, inplace=True)

if 'Event Date' not in df.columns:
    is_synthetic_dates = True
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2024-12-31')
    num_rows = len(df)
    if num_rows > 0:
        monthly_base = pd.date_range(start=start_date, end=end_date, freq='ME')
        dates = np.tile(monthly_base, (num_rows // len(monthly_base)) + 1)[:num_rows]
        df['Event Date'] = dates
    else:
        df['Event Date'] = pd.Series(dtype='datetime64[ns]')

if not is_synthetic_dates and 'Event Date' in df.columns:
    try:
        df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce')
    except Exception:
        df['Event Date'] = pd.to_datetime(df['Event Date'], format='%Y%m%d', errors='coerce')
    
    nan_dates = df['Event Date'].isna().sum()
    if nan_dates > 0:
        st.warning(f"Found {nan_dates} rows with invalid dates in 'Event Date'. These will be excluded where date is crucial.")
        df = df.dropna(subset=['Event Date'])

if 'Event Date' in df.columns and not df['Event Date'].isna().all() and len(df)>0:
    st.write(f"Data spans from {df['Event Date'].min().strftime('%Y-%m-%d')} to {df['Event Date'].max().strftime('%Y-%m-%d')}.")
else:
    st.write("Event dates are not available or fully processed.")


default_numeric_cols = [
    "SigStr1_landed", "Head1_landed", "Body1_landed", "Leg1_landed", "SigStr2_landed", "Head2_landed", "Body2_landed", "Leg2_landed",
    "SigStr2_attempted", "SigStr1_attempted", "Td1_landed", "Td1_attempted", "Td2_landed", "Td2_attempted",
    "KD1", "KD2", "SubAtt1", "SubAtt2", "Ctrl1", "Ctrl2",
    "Reach1", "Reach2", "Height1", "Height2"
]
for col in default_numeric_cols:
    if col not in df.columns:
        df[col] = 0.0

if 'Record1' not in df.columns: df['Record1'] = "0-0-0"
if 'Record2' not in df.columns: df['Record2'] = "0-0-0"
if 'DOB1' not in df.columns: df['DOB1'] = pd.NaT
if 'DOB2' not in df.columns: df['DOB2'] = pd.NaT

# ------------------ Feature Engineering for Both Fighters ------------------
epsilon = 1e-6
df['SVR1'] = (df['SigStr1_landed'] + df['Head1_landed'] + df['Body1_landed'] + df['Leg1_landed']) / (df['Fight_Duration_Min'] + epsilon)
df['SVR2'] = (df['SigStr2_landed'] + df['Head2_landed'] + df['Body2_landed'] + df['Leg2_landed']) / (df['Fight_Duration_Min'] + epsilon)
df['SVR_Diff'] = df['SVR1'] - df['SVR2']

df['DefEff1'] = 1 - (df['SigStr2_landed'] / (df['SigStr2_attempted'] + epsilon))
df['DefEff2'] = 1 - (df['SigStr1_landed'] / (df['SigStr1_attempted'] + epsilon))
df['DefEff_Diff'] = df['DefEff1'] - df['DefEff2']

df['TD_Success1'] = df['Td1_landed'] / (df['Td1_attempted'] + epsilon)
df['TD_Success2'] = df['Td2_landed'] / (df['Td2_attempted'] + epsilon)
df['TD_Diff'] = df['TD_Success1'] - df['TD_Success2']

df['SLpM1'] = df['SigStr1_landed'] / (df['Fight_Duration_Min'] + epsilon)
df['SLpM2'] = df['SigStr2_landed'] / (df['Fight_Duration_Min'] + epsilon)
df['StrAcc1'] = df['SigStr1_landed'] / (df['SigStr1_attempted'] + epsilon)
df['StrAcc2'] = df['SigStr2_landed'] / (df['SigStr2_attempted'] + epsilon)
df['StrAcc_Diff'] = df['StrAcc1'] - df['StrAcc2'] # Added for RF model

df['OAT1'] = df['SLpM1'] * df['StrAcc1']
df['OAT2'] = df['SLpM2'] * df['StrAcc2']
df['OAT_Diff'] = df['OAT1'] - df['OAT2']

df['KDR1'] = df['KD1'] / (df['Fight_Duration_Min'] + epsilon)
df['KDR2'] = df['KD2'] / (df['Fight_Duration_Min'] + epsilon)
df['KDR_Diff'] = df['KDR1'] - df['KDR2']

df['SubAgg1'] = df['SubAtt1'] / (df['Fight_Duration_Min'] + epsilon)
df['SubAgg2'] = df['SubAtt2'] / (df['Fight_Duration_Min'] + epsilon)
df['SubAgg_Diff'] = df['SubAgg1'] - df['SubAgg2']

def parse_record(record):
    if isinstance(record, str):
        try:
            wins, losses, draws = map(int, record.split('-'))
            return wins + losses + draws
        except ValueError: return 0
    return 0
df['TotalFights1'] = df['Record1'].apply(parse_record)
df['TotalFights2'] = df['Record2'].apply(parse_record)
df['Exp_Diff'] = df['TotalFights1'] - df['TotalFights2']

df['DOB1'] = pd.to_datetime(df['DOB1'], errors='coerce')
df['DOB2'] = pd.to_datetime(df['DOB2'], errors='coerce')

if 'Event Date' in df.columns and not df['DOB1'].isna().all() and not df['DOB2'].isna().all():
    df['Age1'] = (df['Event Date'] - df['DOB1']).dt.days / 365.25
    df['Age2'] = (df['Event Date'] - df['DOB2']).dt.days / 365.25
    df['Age_Diff'] = df['Age1'] - df['Age2']
else: 
    df['Age1'], df['Age2'], df['Age_Diff'] = 0.0, 0.0, 0.0
    # if 'Event Date' in df.columns:
    #     st.info("Age could not be calculated for some fighters due to missing DOBs. Using 0 as placeholder.")


df['ReachAdv'] = df['Reach1'] - df['Reach2']
df['HeightAdv'] = df['Height1'] - df['Height2']

alpha = 1.0
df['FCI1'] = df['Ctrl1'] + alpha * df['Td1_landed']
df['FCI2'] = df['Ctrl2'] + alpha * df['Td2_landed']
df['FCI_Diff'] = df['FCI1'] - df['FCI2']

nn_features = [
    'SVR_Diff', 'DefEff_Diff', 'TD_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff',
    'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv', 'FCI_Diff'
]
fighter_perf_features = ['SVR', 'DefEff', 'OAT', 'KDR', 'SubAgg', 'FCI', 'TD_Success', 'SLpM', 'StrAcc',
                         'SigStr_attempted', 'Td_attempted', 'Ctrl', 'KD', 'TotalFights'] # Added more base features for RF
fighter_base_cols_f1 = {ftype: ftype + '1' for ftype in fighter_perf_features}
fighter_base_cols_f2 = {ftype: ftype + '2' for ftype in fighter_perf_features}
# Correcting mapping for _attempted which are already SigStr1_attempted etc.
fighter_base_cols_f1['SigStr_attempted'] = 'SigStr1_attempted'
fighter_base_cols_f2['SigStr_attempted'] = 'SigStr2_attempted'
fighter_base_cols_f1['Td_attempted'] = 'Td1_attempted'
fighter_base_cols_f2['Td_attempted'] = 'Td2_attempted'

fighter_static_attrs_map = {'Age':'Age','Reach':'Reach','Height':'Height','TotalFights':'TotalFights'}

# Ensure all engineered features exist, fill with 0 if any calculation failed due to missing base
for col_prefix in ['SVR', 'DefEff', 'TD_Success', 'SLpM', 'StrAcc', 'OAT', 'KDR', 'SubAgg', 'FCI', 'TotalFights', 'Age']:
    for suffix in ['1', '2']:
        col = col_prefix + suffix
        if col not in df.columns: df[col] = 0.0
for col_suffix in ['_Diff', 'Adv']: # For diff features
    for col_base in ['SVR', 'DefEff', 'TD', 'OAT', 'KDR', 'SubAgg', 'Exp', 'Age', 'Reach', 'Height', 'FCI', 'StrAcc']:
        col = col_base + col_suffix
        if col not in df.columns: df[col] = 0.0


# ------------------ Sidebar Selection ------------------
if 'Fighter1' not in df.columns or 'Fighter2' not in df.columns:
    st.sidebar.error("Fighter1 or Fighter2 columns not found in the data. Cannot select fighters.")
    st.stop()

all_fighters_series = pd.concat([df['Fighter1'], df['Fighter2']]).dropna().unique()
fighters = sorted([f for f in all_fighters_series if f not in ["UnknownFighter1", "UnknownFighter2"]])


if not fighters:
    st.sidebar.error("No valid fighters found in the dataset after filtering unknowns.")
    st.stop()

st.sidebar.header("🤼 Select Fighters")
selected_f1 = st.sidebar.selectbox("Fighter 1", fighters, index=0 if len(fighters)>0 else None, key="f1_select")
available_f2 = [f for f in fighters if f != selected_f1]
selected_f2 = st.sidebar.selectbox("Fighter 2", available_f2, index=0 if len(available_f2)>0 else None, key="f2_select")


@st.cache_data
def get_adjusted_fighter_data(_df_full, fighter_name, base_feature_map_f1, base_feature_map_f2, date_col='Event Date'):
    # Ensure all keys from base_feature_map_f1 are present in _df_full as F1 versions
    # And keys from base_feature_map_f2 are present as F2 versions
    
    cols_f1_needed = list(base_feature_map_f1.values())
    df_as_f1 = _df_full[_df_full['Fighter1'] == fighter_name].copy()
    
    # Check if all needed columns exist for F1 perspective
    missing_cols_f1 = [col for col in cols_f1_needed if col not in df_as_f1.columns]
    if missing_cols_f1:
        #st.warning(f"Missing columns for {fighter_name} as Fighter1: {missing_cols_f1}. Filling with 0.")
        for col in missing_cols_f1: df_as_f1[col] = 0.0
    data_f1 = df_as_f1[[date_col] + cols_f1_needed if date_col in df_as_f1.columns else cols_f1_needed]


    cols_f2_needed = list(base_feature_map_f2.values())
    df_as_f2 = _df_full[_df_full['Fighter2'] == fighter_name].copy()

    missing_cols_f2 = [col for col in cols_f2_needed if col not in df_as_f2.columns]
    if missing_cols_f2:
        #st.warning(f"Missing columns for {fighter_name} as Fighter2: {missing_cols_f2}. Filling with 0.")
        for col in missing_cols_f2: df_as_f2[col] = 0.0
    data_f2 = df_as_f2[[date_col] + cols_f2_needed if date_col in df_as_f2.columns else cols_f2_needed]
    
    # Rename F2 perspective columns to F1 perspective for consistent schema
    rename_map = {v_f2: v_f1 for v_f1, v_f2 in zip(base_feature_map_f1.values(), base_feature_map_f2.values())}
    data_f2 = data_f2.rename(columns=rename_map)
    
    combined_data = pd.concat([data_f1, data_f2], ignore_index=True)
    
    for col in base_feature_map_f1.values(): # Iterate using the target schema (F1 schema)
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')
    
    combined_data = combined_data.dropna(subset=list(base_feature_map_f1.values()), how='all')
    if date_col in combined_data.columns:
        combined_data = combined_data.dropna(subset=[date_col])
        combined_data = combined_data.sort_values(by=date_col).reset_index(drop=True)
    return combined_data

f1_adj_df = get_adjusted_fighter_data(df, selected_f1, fighter_base_cols_f1, fighter_base_cols_f2)
f2_adj_df = get_adjusted_fighter_data(df, selected_f2, fighter_base_cols_f1, fighter_base_cols_f2)

st.title(f"🏆 MMA Fighter Comparison: {selected_f1} vs {selected_f2}")
st.header("🧠 Neural Network Win Prediction")

@st.cache_resource
def build_nn_model(X_train_scaled, y_train, input_dim):
    tf.random.set_seed(42)
    np.random.seed(42)
    model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    return model, history

train_df_nn = df.dropna(subset=nn_features + ['Win'])
X_nn = train_df_nn[nn_features].fillna(0) 
y_nn = train_df_nn['Win']
nn_model = None
scaler_nn = StandardScaler() 

if len(X_nn) < 30 or X_nn.empty:
    st.warning("Insufficient data for neural network training (need at least 30 samples). Predictions may be unreliable or unavailable.")
    if len(X_nn) >= 10: 
        st.info("Training a Logistic Regression model as a fallback for win prediction.")
        X_nn_scaled = scaler_nn.fit_transform(X_nn) 
        fallback_model = LogisticRegression(random_state=42, class_weight='balanced')
        fallback_model.fit(X_nn_scaled, y_nn)
        nn_model = fallback_model 
    else:
        st.error("Cannot make win predictions due to insufficient data for any model.")
else:
    X_nn_scaled = scaler_nn.fit_transform(X_nn) 
    nn_model, history_nn = build_nn_model(X_nn_scaled, y_nn, X_nn_scaled.shape[1])
    
    st.subheader("📈 NN Model Training Performance")
    fig_hist, ax_hist = plt.subplots(1, 2, figsize=(12, 4))
    ax_hist[0].plot(history_nn.history['accuracy'], label='Train Accuracy')
    ax_hist[0].plot(history_nn.history['val_accuracy'], label='Validation Accuracy')
    ax_hist[0].set_title('NN Model Accuracy')
    ax_hist[0].set_ylabel('Accuracy')
    ax_hist[0].set_xlabel('Epoch')
    ax_hist[0].legend()
    
    ax_hist[1].plot(history_nn.history['loss'], label='Train Loss')
    ax_hist[1].plot(history_nn.history['val_loss'], label='Validation Loss')
    ax_hist[1].set_title('NN Model Loss')
    ax_hist[1].set_ylabel('Loss')
    ax_hist[1].set_xlabel('Epoch')
    ax_hist[1].legend()
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close(fig_hist)
    

@st.cache_data
def get_fighter_static_attributes(fighter_name, _df_full, current_date=pd.Timestamp('today')):
    attrs = {'Age': np.nan, 'Reach': np.nan, 'Height': np.nan, 'TotalFights': np.nan}
    fighter_rows = _df_full[(_df_full['Fighter1'] == fighter_name) | (_df_full['Fighter2'] == fighter_name)].copy()
    
    if 'Event Date' in fighter_rows.columns:
        fighter_rows['Event Date'] = pd.to_datetime(fighter_rows['Event Date'], errors='coerce')
        fighter_rows = fighter_rows.sort_values(by='Event Date', ascending=False)
    elif not fighter_rows.empty:
        fighter_rows = fighter_rows.iloc[[0]] 

    if not fighter_rows.empty:
        latest_fight = fighter_rows.iloc[0]
        is_f1 = latest_fight['Fighter1'] == fighter_name
        
        dob_col = 'DOB1' if is_f1 else 'DOB2'
        age_col = 'Age1' if is_f1 else 'Age2' 
        
        if dob_col in latest_fight and pd.notna(latest_fight[dob_col]) and 'Event Date' in latest_fight and pd.notna(latest_fight['Event Date']):
            attrs['Age'] = (latest_fight['Event Date'] - pd.to_datetime(latest_fight[dob_col])).days / 365.25
        elif age_col in latest_fight and pd.notna(latest_fight[age_col]):
             attrs['Age'] = latest_fight[age_col] 
        
        attrs['Reach'] = latest_fight['Reach1' if is_f1 else 'Reach2']
        attrs['Height'] = latest_fight['Height1' if is_f1 else 'Height2']
        attrs['TotalFights'] = latest_fight['TotalFights1' if is_f1 else 'TotalFights2']
        
    for key in attrs:
        if pd.isna(attrs[key]):
            attrs[key] = 0
    return attrs

s1_attrs = get_fighter_static_attributes(selected_f1, df)
s2_attrs = get_fighter_static_attributes(selected_f2, df)

if nn_model and not f1_adj_df.empty and not f2_adj_df.empty and selected_f1 and selected_f2:
    s1_perf_stats = f1_adj_df[[col for col in fighter_base_cols_f1.values() if col in f1_adj_df.columns]].mean().fillna(0)
    s2_perf_stats = f2_adj_df[[col for col in fighter_base_cols_f1.values() if col in f2_adj_df.columns]].mean().fillna(0)


    input_data_nn = {}

    
    # Reconstruct nn_features for prediction:
    for nn_feat_name in nn_features:
        if nn_feat_name.endswith('_Diff'):
            base_metric = nn_feat_name.replace('_Diff', '')
            # Find the F1 schema column name for this base_metric
            # e.g., SVR_Diff -> base_metric SVR -> fighter_base_cols_f1['SVR'] which is 'SVR1'
            f1_col_name = fighter_base_cols_f1.get(base_metric)
            if f1_col_name:
                val1 = s1_perf_stats.get(f1_col_name, 0)
                val2 = s2_perf_stats.get(f1_col_name, 0) # s2_perf_stats also uses F1 schema names
                input_data_nn[nn_feat_name] = val1 - val2
            else: # Handle special diff names like Exp_Diff, Age_Diff etc.
                 pass # These will be handled next
        elif nn_feat_name in ['ReachAdv', 'HeightAdv']: # Already differential by name
            pass # Handled below

    input_data_nn['Exp_Diff'] = s1_attrs['TotalFights'] - s2_attrs['TotalFights']
    input_data_nn['Age_Diff'] = (s1_attrs['Age'] - s2_attrs['Age']) if s1_attrs['Age'] > 0 and s2_attrs['Age'] > 0 else 0.0
    input_data_nn['ReachAdv'] = s1_attrs['Reach'] - s2_attrs['Reach']
    input_data_nn['HeightAdv'] = s1_attrs['Height'] - s2_attrs['Height']

    input_list_ordered_nn = []
    for feature_name in nn_features:
        val = input_data_nn.get(feature_name)
        if val is None:
            #st.warning(f"NN input feature {feature_name} could not be derived for prediction. Using 0.")
            input_list_ordered_nn.append(0)
        else:
            input_list_ordered_nn.append(val)
            
    input_row_nn = np.array(input_list_ordered_nn).reshape(1, -1)
    
    if hasattr(scaler_nn, 'mean_'): 
        input_nn_scaled = scaler_nn.transform(input_row_nn)
    else:
        st.error("NN Scaler not fitted. Win predictions may be inaccurate.")
        input_nn_scaled = input_row_nn 

    win_prob = 0.5 # Default
    if isinstance(nn_model, tf.keras.Model):
        win_prob = nn_model.predict(input_nn_scaled, verbose=0)[0][0]
    elif hasattr(nn_model, 'predict_proba'): 
        win_prob = nn_model.predict_proba(input_nn_scaled)[0][1]

    st.subheader("🎯 Win Probability Prediction")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric(f"{selected_f1} Win Probability", f"{win_prob*100:.1f}%")
        st.progress(float(win_prob))
    with prob_col2:
        st.metric(f"{selected_f2} Win Probability", f"{(1-win_prob)*100:.1f}%")
        st.progress(float(1-win_prob))
    
    predicted_winner = selected_f1 if win_prob > 0.5 else selected_f2
    confidence_val = abs(win_prob - 0.5) * 2
    confidence_text = "High" if confidence_val > 0.7 else "Medium" if confidence_val > 0.4 else "Low"
    st.success(f"🏆 **Predicted Winner (NN): {predicted_winner}** (Confidence: {confidence_text})")

    st.subheader("💬 Match Insights (Based on Average Stats)")
    insights = []
    def compare_metrics_for_insight(val1, val2, f1_name, f2_name, metric_name, higher_is_better=True, threshold_ratio=1.15, unit=""):
        s_val1 = f"{val1:.2f}{unit}"
        s_val2 = f"{val2:.2f}{unit}"
        if higher_is_better:
            if val1 > val2 * threshold_ratio: return f"- {f1_name} has a notably higher {metric_name} ({s_val1} vs {s_val2})."
            if val2 > val1 * threshold_ratio: return f"- {f2_name} has a notably higher {metric_name} ({s_val2} vs {s_val1})."
        else: 
            if val1 < val2 / threshold_ratio: return f"- {f1_name} has a notably lower {metric_name} ({s_val1} vs {s_val2})."
            if val2 < val1 / threshold_ratio: return f"- {f2_name} has a notably lower {metric_name} ({s_val2} vs {s_val1})."
        
        avg_val = (val1 + val2) / 2
        if avg_val > epsilon and abs(val1 - val2) / avg_val < (threshold_ratio - 1) / 2 : 
             return f"- Both fighters are closely matched in {metric_name} ({s_val1} vs {s_val2})."
        return None

    metric_map_insights = {
        'SVR': "Strike Volume Rate", 'DefEff': "Defensive Efficiency", 'TD_Success': "Takedown Success",
        'OAT': "Overall Attack Threat", 'KDR': "Knockdown Rate", 'SubAgg': "Submission Aggression", 'FCI': "Fight Control Index"
    }
    for generic_name, display_name in metric_map_insights.items():
        col_name_f1_schema = fighter_base_cols_f1.get(generic_name)
        if col_name_f1_schema:
            s1_val = s1_perf_stats.get(col_name_f1_schema, 0)
            s2_val = s2_perf_stats.get(col_name_f1_schema, 0) # s2_perf_stats uses F1 schema
            insight = compare_metrics_for_insight(s1_val, s2_val, selected_f1, selected_f2, display_name)
            if insight: insights.append(insight)
    
    if abs(s1_attrs['Age'] - s2_attrs['Age']) > 3 and s1_attrs['Age'] > 0 and s2_attrs['Age'] > 0:
        older_fighter = selected_f1 if s1_attrs['Age'] > s2_attrs['Age'] else selected_f2
        younger_fighter = selected_f2 if s1_attrs['Age'] > s2_attrs['Age'] else selected_f1
        insights.append(f"- {older_fighter} ({s1_attrs['Age']:.1f} yrs) is significantly older than {younger_fighter} ({s2_attrs['Age']:.1f} yrs).")
    if abs(s1_attrs['Reach'] - s2_attrs['Reach']) > 5: 
        reach_adv_fighter = selected_f1 if s1_attrs['Reach'] > s2_attrs['Reach'] else selected_f2
        insights.append(f"- {reach_adv_fighter} has a notable reach advantage.")

    if not insights: insights.append("- Fighters appear closely matched on these key metrics, or data is limited for deep insights.")
    for insight in insights: st.write(insight)

    st.subheader("🔑 Keys to Victory (Potential based on NN Prediction)")
    keys_f1, keys_f2 = [], []
    if predicted_winner == selected_f1:
        if input_data_nn.get('TD_Diff', 0) > 0.1: keys_f1.append("Exploit takedown advantage.")
        if input_data_nn.get('SVR_Diff', 0) > 5: keys_f1.append("Maintain high striking volume.")
        if input_data_nn.get('ReachAdv', 0) > 5: keys_f1.append("Utilize reach to control distance.")
        if not keys_f1: keys_f1.append("Stick to the game plan; fight is predicted to be close or strengths are balanced.")
        st.markdown(f"**For {selected_f1}:**")
        for key in keys_f1: st.markdown(f"  - {key}")
    else:
        if input_data_nn.get('TD_Diff', 0) < -0.1: keys_f2.append("Exploit takedown advantage.")
        if input_data_nn.get('SVR_Diff', 0) < -5: keys_f2.append("Maintain high striking volume.")
        if input_data_nn.get('ReachAdv', 0) < -5: keys_f2.append("Utilize reach to control distance.")
        if not keys_f2: keys_f2.append("Stick to the game plan; fight is predicted to be close or strengths are balanced.")
        st.markdown(f"**For {selected_f2}:**")
        for key in keys_f2: st.markdown(f"  - {key}")

elif not selected_f1 or not selected_f2:
    st.info("Please select two different fighters for win prediction.")
else:
    st.warning("Insufficient data for selected fighters or NN model not trained; cannot compute win prediction or full insights.")


# ------------------ Style Clustering ------------------
st.header("🧬 Fighter Style Clustering")
f1_style, f2_style = "Style Undetermined", "Style Undetermined"
df_cluster_input = df[['SVR1', 'FCI1']].copy()
for col in ['SVR1', 'FCI1']:
    df_cluster_input[col] = pd.to_numeric(df_cluster_input[col], errors='coerce')
df_cluster_input = df_cluster_input.dropna()

if not df_cluster_input.empty and len(df_cluster_input) >= 4: # Kmeans needs at least n_clusters samples
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(df_cluster_input)
    num_clusters = min(4, len(df_cluster_input)) # Ensure n_clusters <= n_samples
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_cluster_scaled)
    df_clustered = df_cluster_input.copy()
    df_clustered['Cluster'] = clusters
    original_centers_cluster = scaler_cluster.inverse_transform(kmeans.cluster_centers_)

    cluster_styles = {}
    # Check if we have enough clusters to define by median comparison
    if num_clusters >=2 : # Need at least 2 points to define a median robustly
        svr_median_cluster = np.median(original_centers_cluster[:, 0])
        fci_median_cluster = np.median(original_centers_cluster[:, 1])
        for i, center in enumerate(original_centers_cluster):
            style = ""
            if center[0] > svr_median_cluster and center[1] > fci_median_cluster: style = "Pressure Fighter"
            elif center[0] > svr_median_cluster and center[1] <= fci_median_cluster: style = "Volume Striker"
            elif center[0] <= svr_median_cluster and center[1] > fci_median_cluster: style = "Ground Specialist"
            else: style = "Tactical Counter-Fighter"
            cluster_styles[i] = style
    else: # Fallback for very few clusters
        for i in range(num_clusters): cluster_styles[i] = f"Cluster {i+1} Profile"

    df_clustered['Style'] = df_clustered['Cluster'].map(cluster_styles)

    st.subheader("🎨 Fighter Clusters & Styles (based on Fighter1 stats per fight)")
    fig_cluster, ax_cluster = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x='SVR1', y='FCI1', hue='Style', data=df_clustered, palette='Set2', ax=ax_cluster, s=60, alpha=0.7)

    f1_avg_svr, f1_avg_fci, f2_avg_svr, f2_avg_fci = np.nan, np.nan, np.nan, np.nan
    if not f1_adj_df.empty:
        f1_avg_svr = f1_adj_df[fighter_base_cols_f1['SVR']].mean() if fighter_base_cols_f1['SVR'] in f1_adj_df else np.nan
        f1_avg_fci = f1_adj_df[fighter_base_cols_f1['FCI']].mean() if fighter_base_cols_f1['FCI'] in f1_adj_df else np.nan
        if pd.notna(f1_avg_svr) and pd.notna(f1_avg_fci):
             ax_cluster.scatter(f1_avg_svr, f1_avg_fci, color='blue', s=250, label=f"{selected_f1} (Avg)", edgecolors='black', marker='o', linewidth=1.5, zorder=5)
    if not f2_adj_df.empty:
        f2_avg_svr = f2_adj_df[fighter_base_cols_f1['SVR']].mean() if fighter_base_cols_f1['SVR'] in f2_adj_df else np.nan
        f2_avg_fci = f2_adj_df[fighter_base_cols_f1['FCI']].mean() if fighter_base_cols_f1['FCI'] in f2_adj_df else np.nan
        if pd.notna(f2_avg_svr) and pd.notna(f2_avg_fci):
            ax_cluster.scatter(f2_avg_svr, f2_avg_fci, color='darkorange', s=250, label=f"{selected_f2} (Avg)", edgecolors='black', marker='s', linewidth=1.5, zorder=5)

    for i, center in enumerate(original_centers_cluster):
        ax_cluster.scatter(center[0], center[1], color='red', s=200, marker='X', label=f"Center: {cluster_styles.get(i, 'N/A')}")
        ax_cluster.text(center[0]*1.02, center[1]*1.02, cluster_styles.get(i, 'N/A'), fontsize=9, ha='left', va='bottom', 
                        bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))

    plt.title("Fighter Style Clustering (Strike Volume Rate vs Fight Control Index)")
    plt.xlabel("Strike Volume Rate (SVR1 from fights)")
    plt.ylabel("Fight Control Index (FCI1 from fights)")
    handles, labels = ax_cluster.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels in legend
    ax_cluster.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.5), title="Legend")
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust layout to make space for legend
    st.pyplot(fig_cluster)
    plt.close(fig_cluster)

    def get_fighter_style(avg_svr, avg_fci, _cluster_centers_orig, _styles_map):
        if pd.isna(avg_svr) or pd.isna(avg_fci): return "Style Undetermined (Missing Avg Stats)"
        if not _cluster_centers_orig.size: return "Style Undetermined (No Clusters)"
        distances = [np.sqrt((avg_svr - center[0])**2 + (avg_fci - center[1])**2) for center in _cluster_centers_orig]
        closest_cluster_idx = np.argmin(distances)
        return _styles_map.get(closest_cluster_idx, "Style Undetermined (Cluster Map Error)")

    if pd.notna(f1_avg_svr): f1_style = get_fighter_style(f1_avg_svr, f1_avg_fci, original_centers_cluster, cluster_styles)
    if pd.notna(f2_avg_svr): f2_style = get_fighter_style(f2_avg_svr, f2_avg_fci, original_centers_cluster, cluster_styles)
    
    st.write(f"**{selected_f1}'s Estimated Fighting Style:** {f1_style}")
    st.write(f"**{selected_f2}'s Estimated Fighting Style:** {f2_style}")

    st.subheader("📊 Cluster Profile Averages")
    cluster_profile_df = pd.DataFrame(original_centers_cluster, columns=['Avg SVR', 'Avg FCI'])
    cluster_profile_df['Style'] = cluster_profile_df.index.map(cluster_styles)
    st.dataframe(cluster_profile_df[['Style', 'Avg SVR', 'Avg FCI']].style.format({'Avg SVR': "{:.2f}", 'Avg FCI': "{:.2f}"}))
else:
    st.warning("Insufficient data for style clustering (need at least 4 data points with SVR1 and FCI1). Styles will be undetermined.")


# ------------------ Tale of the Tape & Fighter Profiles ------------------
st.header("🥊 Tale of the Tape & Fighter Profiles")

@st.cache_data
def get_last_n_fights_outcomes(fighter_name, _df_full, n=5):
    fighter_df_rows = _df_full[
        (_df_full['Fighter1'] == fighter_name) | (_df_full['Fighter2'] == fighter_name)
    ].copy()
    
    if 'Event Date' in fighter_df_rows.columns and not fighter_df_rows['Event Date'].isna().all() :
        fighter_df_rows['Event Date'] = pd.to_datetime(fighter_df_rows['Event Date'], errors='coerce')
        fighter_df_rows = fighter_df_rows.sort_values(by='Event Date', ascending=False)
    elif not fighter_df_rows.empty: 
        pass 
    else: 
        return "N/A (No Fight History)"

    outcomes = []
    for _, row in fighter_df_rows.head(n).iterrows():
        winner_col = row.get('winner')
        if pd.isna(winner_col) or str(winner_col).lower() in ['draw', 'nc', 'unknown', 'no contest']:
            outcomes.append("D/NC")
        elif str(winner_col).lower() == fighter_name.lower():
            outcomes.append("✔️ W")
        else:
            outcomes.append("❌ L")
    return ", ".join(outcomes) if outcomes else "No Fights Recorded"

@st.cache_data
def get_latest_record(fighter_name, _df_full):
    fighter_rows = _df_full[(_df_full['Fighter1'] == fighter_name) | (_df_full['Fighter2'] == fighter_name)].copy()
    if 'Event Date' in fighter_rows.columns and not fighter_rows['Event Date'].isna().all():
        fighter_rows['Event Date'] = pd.to_datetime(fighter_rows['Event Date'], errors='coerce')
        fighter_rows = fighter_rows.sort_values(by='Event Date', ascending=False)
    
    if not fighter_rows.empty:
        latest_fight = fighter_rows.iloc[0]
        is_f1 = latest_fight['Fighter1'] == fighter_name
        record_col = 'Record1' if is_f1 else 'Record2'
        return latest_fight.get(record_col, "N/A")
    return "N/A"

f1_record = get_latest_record(selected_f1, df)
f2_record = get_latest_record(selected_f2, df)
s1_last5 = get_last_n_fights_outcomes(selected_f1, df)
s2_last5 = get_last_n_fights_outcomes(selected_f2, df)

profile_col1, profile_col2 = st.columns(2)
with profile_col1:
    st.subheader(f"👤 {selected_f1}")
    st.markdown(f"**Est. Style:** {f1_style}")
    st.markdown(f"**Last 5 Fights:** {s1_last5}")
with profile_col2:
    st.subheader(f"👤 {selected_f2}")
    st.markdown(f"**Est. Style:** {f2_style}")
    st.markdown(f"**Last 5 Fights:** {s2_last5}")


# ------------------ Strength & Weakness Comparison ------------------
st.header("💪 Strength & Weakness Comparison")
if not f1_adj_df.empty and not f2_adj_df.empty and 's1_perf_stats' in locals() and 's2_perf_stats' in locals():
    comp_features_generic = fighter_perf_features[:6] 
    comp_features_f1_schema = [fighter_base_cols_f1[f] for f in comp_features_generic if f in fighter_base_cols_f1]

    mean_f1_comp = s1_perf_stats[comp_features_f1_schema].fillna(0)
    mean_f2_comp = s2_perf_stats[comp_features_f1_schema].fillna(0) # s2_perf_stats uses F1 schema
    
    comparison_values = mean_f1_comp - mean_f2_comp
    # Filter comp_features_generic to match available comparison_values indices
    valid_comp_features_generic = [f for f, col_f1 in zip(comp_features_generic, comp_features_f1_schema) if col_f1 in comparison_values.index]
    
    comparison_df = pd.DataFrame({'Difference': comparison_values.values, 'Feature': valid_comp_features_generic})


    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    colors = ["cornflowerblue" if x > 0 else "salmon" for x in comparison_df['Difference']]
    bars = sns.barplot(x='Difference', y='Feature', data=comparison_df, palette=colors, ax=ax_comp, orient='h')
    
    for i, bar_patch in enumerate(bars.patches): # Renamed bar to bar_patch
        value = comparison_df['Difference'].iloc[i]
        ax_comp.text(bar_patch.get_width(), bar_patch.get_y() + bar_patch.get_height()/2, f'{value:.2f}', 
                     ha='left' if value >= 0 else 'right', va='center', color='black')
    plt.xlabel(f"Advantage ({selected_f1} vs {selected_f2})")
    plt.title("Key Performance Metric Comparison (Average Career Stats)")
    plt.axvline(0, color='grey', linewidth=0.8)
    st.pyplot(fig_comp)
    plt.close(fig_comp)

    st.subheader(f"🕸️ Fighter Attributes Radar")
    radar_features_generic = ['SVR', 'DefEff', 'OAT', 'KDR', 'SubAgg', 'TD_Success'] 
    
    stats_f1_radar_values = [s1_perf_stats.get(fighter_base_cols_f1.get(feat), 0) for feat in radar_features_generic]
    stats_f2_radar_values = [s2_perf_stats.get(fighter_base_cols_f1.get(feat), 0) for feat in radar_features_generic]
        
    plot_df_radar = pd.DataFrame({
        'Attribute': radar_features_generic,
        selected_f1: stats_f1_radar_values,
        selected_f2: stats_f2_radar_values
    }).set_index('Attribute')

    normalized_plot_df_radar = plot_df_radar.copy()
    for attr_radar in radar_features_generic: # Renamed attr to attr_radar
        max_val_radar = plot_df_radar.loc[attr_radar].max() # Renamed max_val
        if max_val_radar > epsilon: 
            normalized_plot_df_radar.loc[attr_radar] = (plot_df_radar.loc[attr_radar] / max_val_radar) * 100 
        else:
            normalized_plot_df_radar.loc[attr_radar] = 0
    
    labels_radar = normalized_plot_df_radar.index.to_numpy()
    n_vars_radar = len(labels_radar)
    angles_radar = np.linspace(0, 2 * np.pi, n_vars_radar, endpoint=False).tolist()
    angles_radar += angles_radar[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    def add_to_radar(fighter_name_radar, values_radar, angle_list_radar, color_radar, ax_r):
        data_radar_plot = np.concatenate((values_radar, values_radar[:1])) # Renamed data_radar
        ax_r.plot(angle_list_radar, data_radar_plot, linewidth=2, linestyle='solid', label=fighter_name_radar, color=color_radar, zorder=3)
        ax_r.fill(angle_list_radar, data_radar_plot, alpha=0.25, color=color_radar, zorder=2)

    add_to_radar(selected_f1, normalized_plot_df_radar[selected_f1].values, angles_radar, 'cornflowerblue', ax_radar)
    add_to_radar(selected_f2, normalized_plot_df_radar[selected_f2].values, angles_radar, 'salmon', ax_radar)
    
    ax_radar.set_xticks(angles_radar[:-1])
    ax_radar.set_xticklabels(labels_radar)
    ax_radar.set_yticks(np.arange(0, 101, 20)) 
    ax_radar.set_yticklabels([f"{i}%" for i in np.arange(0, 101, 20)])
    ax_radar.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax_radar.xaxis.grid(True, linestyle='--', alpha=0.7)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Normalized Performance Radar (Max of pair = 100%)", size=14, y=1.1, color='gray')
    st.pyplot(fig_radar)
    plt.close(fig_radar)
    st.caption("Note: Radar attributes are normalized. For each attribute, the fighter with the higher value is scaled to 100%, and the other fighter is scaled proportionally. This helps visualize relative strengths.")

else:
    st.write(f"Not enough data for Strength & Weakness Comparison. {selected_f1} fights: {len(f1_adj_df)}, {selected_f2} fights: {len(f2_adj_df)}")


# ------------------ Trend Over Time ------------------
st.header("📅 Performance Trend Over Time")
trend_features_generic = ['SVR', 'KDR', 'TD_Success']
trend_features_display_names = {
    'SVR': 'Strike Volume Rate', 'KDR': 'Knockdown Rate', 'TD_Success': 'Takedown Success Rate'
}

if 'Event Date' not in df.columns or df['Event Date'].isna().all():
    st.warning("Event Date column is missing or empty. Cannot display performance trends.")
else:
    # df['Event Date'] = pd.to_datetime(df['Event Date'], errors='coerce') # Already done earlier
    
    if selected_f1 and not f1_adj_df.empty and 'Event Date' in f1_adj_df.columns:
        st.subheader(f"📈 Trends for {selected_f1}")
        fig_trend_f1, axes_f1 = plt.subplots(len(trend_features_generic), 1, figsize=(10, 2.5 * len(trend_features_generic)), sharex=True)
        if len(trend_features_generic) == 1: axes_f1 = [axes_f1] # Ensure axes_f1 is iterable

        for ax_trend_f1, feature_generic_name_trend in zip(axes_f1, trend_features_generic): # Renamed ax, feature_generic_name
            feature_f1_schema_trend = fighter_base_cols_f1.get(feature_generic_name_trend) # Renamed feature_f1_schema
            if feature_f1_schema_trend and feature_f1_schema_trend in f1_adj_df.columns:
                plot_data_f1_trend = f1_adj_df.dropna(subset=['Event Date', feature_f1_schema_trend]) # Renamed plot_data_f1
                if not plot_data_f1_trend.empty:
                    ax_trend_f1.plot(plot_data_f1_trend['Event Date'], plot_data_f1_trend[feature_f1_schema_trend], marker='o', linestyle='-', label='Per Fight', color='cornflowerblue')
                    if len(plot_data_f1_trend) >= 3:
                        ax_trend_f1.plot(plot_data_f1_trend['Event Date'], plot_data_f1_trend[feature_f1_schema_trend].rolling(window=3, center=True, min_periods=1).mean(),
                                linestyle='--', label='3-Fight Avg', color='darkblue')
                    ax_trend_f1.legend()
                ax_trend_f1.set_ylabel(trend_features_display_names.get(feature_generic_name_trend, feature_generic_name_trend))
                ax_trend_f1.grid(True, linestyle=':', alpha=0.7)
        plt.xlabel("Event Date")
        fig_trend_f1.suptitle(f"Performance Trends for {selected_f1}", fontsize=14, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig_trend_f1)
        plt.close(fig_trend_f1)
    else:
        st.write(f"Not enough data or {selected_f1} not selected/valid for trend analysis.")

    if selected_f2 and not f2_adj_df.empty and 'Event Date' in f2_adj_df.columns:
        st.subheader(f"📈 Trends for {selected_f2}")
        fig_trend_f2, axes_f2 = plt.subplots(len(trend_features_generic), 1, figsize=(10, 2.5 * len(trend_features_generic)), sharex=True)
        if len(trend_features_generic) == 1: axes_f2 = [axes_f2] # Ensure axes_f2 is iterable

        for ax_trend_f2, feature_generic_name_trend2 in zip(axes_f2, trend_features_generic): # Renamed ax, feature_generic_name
            feature_f1_schema_trend2 = fighter_base_cols_f1.get(feature_generic_name_trend2) # Renamed feature_f1_schema
            if feature_f1_schema_trend2 and feature_f1_schema_trend2 in f2_adj_df.columns:
                plot_data_f2_trend = f2_adj_df.dropna(subset=['Event Date', feature_f1_schema_trend2]) # Renamed plot_data_f2
                if not plot_data_f2_trend.empty:
                    ax_trend_f2.plot(plot_data_f2_trend['Event Date'], plot_data_f2_trend[feature_f1_schema_trend2], marker='o', linestyle='-', label='Per Fight', color='salmon')
                    if len(plot_data_f2_trend) >= 3:
                        ax_trend_f2.plot(plot_data_f2_trend['Event Date'], plot_data_f2_trend[feature_f1_schema_trend2].rolling(window=3, center=True, min_periods=1).mean(),
                                linestyle='--', label='3-Fight Avg', color='darkred')
                    ax_trend_f2.legend()
                ax_trend_f2.set_ylabel(trend_features_display_names.get(feature_generic_name_trend2, feature_generic_name_trend2))
                ax_trend_f2.grid(True, linestyle=':', alpha=0.7)
        plt.xlabel("Event Date")
        fig_trend_f2.suptitle(f"Performance Trends for {selected_f2}", fontsize=14, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        st.pyplot(fig_trend_f2)
        plt.close(fig_trend_f2)
    else:
        st.write(f"Not enough data or {selected_f2} not selected/valid for trend analysis.")


# ------------------ Strike Prediction Model (Random Forest) ------------------
st.header("🎯 Strike Count Prediction (Random Forest)")

actual_features_for_rf_models = []
for pfix in ['StrAcc', 'SigStr_attempted', 'TD_Success', 'Td_attempted', 'SubAtt', 'Ctrl', 'KD', 
             'SVR', 'DefEff', 'OAT', 'KDR', 'SubAgg', 'FCI', 
             'Height', 'Reach', 'Age', 'TotalFights']:
    for suffix in ['1', '2']:
        actual_features_for_rf_models.append(pfix + suffix)

target_cols_rf = [
    'SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
    'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed'
]

# Ensure all required columns exist in df, fill with 0 if not
for col_list in [actual_features_for_rf_models, target_cols_rf]:
    for col in col_list:
        if col not in df.columns:            
            df[col] = 0.0

df_rf = df[actual_features_for_rf_models + target_cols_rf].copy()
df_rf = df_rf.dropna().reset_index(drop=True)

rf_models = None
scaler_rf = None
X_rf_test_scaled = None # To store for plotting
y_rf_test = None       # To store for plotting
predictions_rf_all = {} # To store for plotting

@st.cache_resource
def train_strike_prediction_models(_X_train_scaled, _y_train, _feature_names, _target_names):
    trained_models = {}
    for target_col_rf in _target_names: # Renamed target to target_col_rf
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(_X_train_scaled, _y_train[target_col_rf])
        trained_models[target_col_rf] = rf
    return trained_models

if len(df_rf) >= 20: # Need enough data for a meaningful train/test split
    X_rf = df_rf[actual_features_for_rf_models]
    y_rf = df_rf[target_cols_rf]
    
    X_rf_train, X_rf_test, y_rf_train, y_rf_test_local = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42) # y_rf_test_local to avoid global overwrite
    y_rf_test = y_rf_test_local # Assign to global for plotting after successful split

    scaler_rf = StandardScaler()
    X_rf_train_scaled = scaler_rf.fit_transform(X_rf_train)
    X_rf_test_scaled = scaler_rf.transform(X_rf_test) # Assign to global

    rf_models = train_strike_prediction_models(X_rf_train_scaled, y_rf_train, actual_features_for_rf_models, target_cols_rf)

    st.subheader("📈 Random Forest Model Evaluation (Test Set)")
    eval_metrics = []
    for target_col_eval in target_cols_rf: # Renamed target to target_col_eval
        if rf_models and target_col_eval in rf_models:
            model_rf_eval = rf_models[target_col_eval] # Renamed model_rf to model_rf_eval
            predictions_rf = model_rf_eval.predict(X_rf_test_scaled)
            predictions_rf_all[target_col_eval] = predictions_rf # Store for plotting

            mae = mean_absolute_error(y_rf_test[target_col_eval], predictions_rf)
            mse = mean_squared_error(y_rf_test[target_col_eval], predictions_rf)
            r2 = r2_score(y_rf_test[target_col_eval], predictions_rf)
            eval_metrics.append({'Target': target_col_eval, 'MAE': mae, 'MSE': mse, 'R²': r2})
        else:
            st.warning(f"Model for target {target_col_eval} not found or not trained.")

    if eval_metrics:
        eval_df = pd.DataFrame(eval_metrics)
        st.dataframe(eval_df.style.format({'MAE': "{:.2f}", 'MSE': "{:.2f}", 'R²': "{:.2f}"}))

    # Strike Prediction for Selected Fighters
    if rf_models and scaler_rf and selected_f1 and selected_f2 and not f1_adj_df.empty and not f2_adj_df.empty and 's1_perf_stats' in locals() and 's2_perf_stats' in locals():
        st.subheader(f"🔮 Predicted Strikes for {selected_f1} vs {selected_f2}")
        input_data_rf_list = []              
        
        temp_input_dict = {}
        for pfix in ['StrAcc', 'SigStr_attempted', 'TD_Success', 'Td_attempted', 'SubAtt', 'Ctrl', 'KD', 
                     'SVR', 'DefEff', 'OAT', 'KDR', 'SubAgg', 'FCI']:
            col_f1_schema = fighter_base_cols_f1.get(pfix) # e.g. SVR -> SVR1
            if col_f1_schema:
                temp_input_dict[pfix + '1'] = s1_perf_stats.get(col_f1_schema, 0)
                temp_input_dict[pfix + '2'] = s2_perf_stats.get(col_f1_schema, 0)
        
        for static_attr_key, static_attr_val in fighter_static_attrs_map.items(): # Age, Height, Reach, TotalFights
            temp_input_dict[static_attr_val + '1'] = s1_attrs.get(static_attr_key, 0)
            temp_input_dict[static_attr_val + '2'] = s2_attrs.get(static_attr_key, 0)

        # Construct the ordered list for prediction
        for feature_name_rf_pred in actual_features_for_rf_models: # Renamed feature_name to feature_name_rf_pred
            input_data_rf_list.append(temp_input_dict.get(feature_name_rf_pred, 0))

        input_row_rf = np.array(input_data_rf_list).reshape(1, -1)
        input_row_rf_scaled = scaler_rf.transform(input_row_rf)

        predicted_strikes_output = {}
        for target_col_pred_rf in target_cols_rf: # Renamed target to target_col_pred_rf
            if target_col_pred_rf in rf_models:
                pred_val_rf = rf_models[target_col_pred_rf].predict(input_row_rf_scaled)[0]
                predicted_strikes_output[target_col_pred_rf] = max(0, pred_val_rf) # Ensure non-negative

        # Display predicted strikes in two columns
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.markdown(f"**{selected_f1} (Predicted)**")
            st.metric("Total Sig. Strikes Landed", f"{predicted_strikes_output.get('SigStr1_landed', 0):.1f}")
            st.metric("Head Strikes Landed", f"{predicted_strikes_output.get('Head1_landed', 0):.1f}")
            st.metric("Body Strikes Landed", f"{predicted_strikes_output.get('Body1_landed', 0):.1f}")
            st.metric("Leg Strikes Landed", f"{predicted_strikes_output.get('Leg1_landed', 0):.1f}")
        with pred_col2:
            st.markdown(f"**{selected_f2} (Predicted)**")
            st.metric("Total Sig. Strikes Landed", f"{predicted_strikes_output.get('SigStr2_landed', 0):.1f}")
            st.metric("Head Strikes Landed", f"{predicted_strikes_output.get('Head2_landed', 0):.1f}")
            st.metric("Body Strikes Landed", f"{predicted_strikes_output.get('Body2_landed', 0):.1f}")
            st.metric("Leg Strikes Landed", f"{predicted_strikes_output.get('Leg2_landed', 0):.1f}")
    else:
        st.info("Select two fighters with sufficient data to predict strike counts.")


    # Plotting for Random Forest - only if data was available for testing
    if X_rf_test_scaled is not None and y_rf_test is not None and predictions_rf_all:
        st.subheader("📊 Random Forest Model Performance Plots")

        if st.checkbox("Show Learning Curves (RF)"):
            with st.spinner("Generating Learning Curves..."):
                for target_lc_rf in target_cols_rf: # Renamed target to target_lc_rf
                    if target_lc_rf in rf_models:
                        estimator_lc_rf = rf_models[target_lc_rf] # Renamed estimator to estimator_lc_rf
                        train_sizes, train_scores, test_scores = learning_curve(
                            estimator_lc_rf, X_rf_train_scaled, y_rf_train[target_lc_rf], cv=3, n_jobs=-1, # Reduced CV for speed
                            train_sizes=np.linspace(0.1, 1.0, 5), scoring='neg_mean_absolute_error', random_state=42) # Reduced train_sizes
                        train_scores_mean = -train_scores.mean(axis=1)
                        test_scores_mean = -test_scores.mean(axis=1)
                        
                        fig_lc, ax_lc = plt.subplots(figsize=(8, 5)) # Renamed fig, ax
                        ax_lc.plot(train_sizes, train_scores_mean, label='Training MAE')
                        ax_lc.plot(train_sizes, test_scores_mean, label='Validation MAE')
                        ax_lc.set_title(f'Learning Curve: {target_lc_rf}')
                        ax_lc.set_xlabel('Training Examples')
                        ax_lc.set_ylabel('Mean Absolute Error')
                        ax_lc.legend(loc='best')
                        ax_lc.grid(True)
                        st.pyplot(fig_lc)
                        plt.close(fig_lc)
        
        if st.checkbox("Show Predicted vs Actual Plot (RF)"):
            fig_pvsa, ax_pvsa = plt.subplots(figsize=(10, 6)) # Renamed fig, ax
            min_val_pvsa, max_val_pvsa = float('inf'), float('-inf') # Renamed min_val, max_val

            for target_pvsa_rf in target_cols_rf: # Renamed target to target_pvsa_rf
                if target_pvsa_rf in predictions_rf_all and target_pvsa_rf in y_rf_test.columns:
                    actual_pvsa = y_rf_test[target_pvsa_rf] # Renamed actual
                    predicted_pvsa = predictions_rf_all[target_pvsa_rf] # Renamed predicted
                    ax_pvsa.scatter(actual_pvsa, predicted_pvsa, alpha=0.5, label=target_pvsa_rf)
                    min_val_pvsa = min(min_val_pvsa, actual_pvsa.min(), predicted_pvsa.min())
                    max_val_pvsa = max(max_val_pvsa, actual_pvsa.max(), predicted_pvsa.max())
            
            if np.isfinite(min_val_pvsa) and np.isfinite(max_val_pvsa):
                 ax_pvsa.plot([min_val_pvsa, max_val_pvsa], [min_val_pvsa, max_val_pvsa], 'r--')
            ax_pvsa.set_xlabel('Actual Strikes')
            ax_pvsa.set_ylabel('Predicted Strikes')
            ax_pvsa.set_title('Predicted vs Actual Strikes (All Targets)')
            ax_pvsa.legend()
            ax_pvsa.grid(True)
            st.pyplot(fig_pvsa)
            plt.close(fig_pvsa)

        if st.checkbox("Show Residual Plot (RF)"):
            fig_res, ax_res = plt.subplots(figsize=(10, 6)) # Renamed fig, ax
            for target_res_rf in target_cols_rf: # Renamed target to target_res_rf
                 if target_res_rf in predictions_rf_all and target_res_rf in y_rf_test.columns:
                    residuals_rf = y_rf_test[target_res_rf] - predictions_rf_all[target_res_rf] # Renamed residuals
                    sns.histplot(residuals_rf, kde=True, label=target_res_rf, ax=ax_res, alpha=0.5, stat="density")
            ax_res.set_title('Residual Distribution (All Targets)')
            ax_res.set_xlabel('Residuals (Actual - Predicted)')
            ax_res.legend()
            ax_res.grid(True)
            st.pyplot(fig_res)
            plt.close(fig_res)

        if st.checkbox("Show Feature Importances (RF)"):
            # Combined plot for top N features across all models, or select one model
            # For simplicity, showing for the first target model (e.g., SigStr1_landed)
            # Or average importance - let's do average for now if multiple models exist.
            
            if rf_models:
                importances_rf_list = [] # Renamed importances_rf
                for target_fi_rf, model_fi_rf in rf_models.items(): # Renamed target, model
                    importances_rf_list.append(pd.Series(model_fi_rf.feature_importances_, index=actual_features_for_rf_models))
                
                if importances_rf_list:
                    avg_importances_rf = pd.concat(importances_rf_list, axis=1).mean(axis=1) # Renamed avg_importances
                    top_n_features = 20 
                    avg_importances_rf = avg_importances_rf.sort_values(ascending=False)[:top_n_features]

                    fig_fi, ax_fi = plt.subplots(figsize=(12, 8)) # Renamed fig, ax
                    avg_importances_rf.plot(kind='bar', ax=ax_fi)
                    ax_fi.set_title(f'Average Top {top_n_features} Feature Importances (RF Strike Prediction)')
                    ax_fi.set_xlabel('Features')
                    ax_fi.set_ylabel('Average Importance')
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig_fi)
                    plt.close(fig_fi)


else:
    st.warning("Not enough data (need at least 20 valid records after processing) to train Random Forest strike prediction models.")


st.markdown("---")