# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
# # from sklearn.ensemble import GradientBoostingRegressor
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from tqdm import tqdm
# # import warnings

# # # Suppress warnings for cleaner output
# # warnings.filterwarnings('ignore')

# # # ------------------ Load Data ------------------
# # def load_data():
# #     df = pd.read_csv("C:/Users/abhi0/OneDrive/Desktop/IIITH/SMAI/PROJECT_clean_dataset/preprocessed_fight_fighters_data_dates_formatted.csv")
# #     df = df.dropna(subset=["winner", "SigStr1_landed", "Ctrl1", "KD1", "Td1_landed"])
# #     df["Win"] = (df['winner'].str.lower() == df['Fighter1'].str.lower()).astype(int)
# #     return df

# # df = load_data()

# # # Add Fight_Duration_Min if not present (assuming 15 minutes for 3-round fights)
# # if 'Fight_Duration_Min' not in df.columns:
# #     df['Fight_Duration_Min'] = 15.0  # Placeholder; adjust based on actual data

# # # Check for event date column and standardize its name
# # possible_date_columns = ['Event Date', 'event_date', 'Date', 'date', 'event date']
# # event_date_col = None
# # for col in possible_date_columns:
# #     if col in df.columns:
# #         event_date_col = col
# #         break

# # # Flag to track if dates are synthetic
# # is_synthetic_dates = False

# # if event_date_col:
# #     df.rename(columns={event_date_col: 'Event Date'}, inplace=True)
# # else:
# #     # Create synthetic Event Date
# #     is_synthetic_dates = True
# #     start_date = pd.Timestamp('2020-01-01')
# #     end_date = pd.Timestamp('2024-12-31')
# #     max_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
# #     num_rows = len(df)
# #     if num_rows <= max_months:
# #         dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=num_rows)
# #     else:
# #         dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=max_months)
# #         dates = np.tile(dates, (num_rows // max_months) + 1)[:num_rows]
# #     df['Event Date'] = dates

# # # Convert numeric Event Date to datetime
# # if not is_synthetic_dates:
# #     df['Event Date'] = df['Event Date'].astype(str)
# #     df['Event Date'] = pd.to_datetime(df['Event Date'], format='%Y%m%d', errors='coerce')

# # # Warn about NaN dates
# # nan_dates = df['Event Date'].isna().sum()
# # if nan_dates > 0:
# #     print(f"Warning: Found {nan_dates} rows with invalid dates in 'Event Date'. These will be excluded from the timeline.")

# # # ------------------ Create Round-Wise Data ------------------
# # # Assume 3 rounds per fight, divide fight-level metrics by 3
# # num_rounds = 3
# # round_dfs = []
# # for round_num in range(1, num_rounds + 1):
# #     round_df = df.copy()
# #     round_df['Round'] = round_num
# #     # Divide fight-level metrics by number of rounds
# #     for col in ['SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
# #                 'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed',
# #                 'Ctrl1', 'Ctrl2', 'KD1', 'KD2', 'Td1_landed', 'Td2_landed',
# #                 'SubAtt1', 'SubAtt2', 'SigStr1_attempted', 'SigStr2_attempted',
# #                 'Head1_attempted', 'Head2_attempted', 'Body1_attempted', 'Body2_attempted',
# #                 'Leg1_attempted', 'Leg2_attempted', 'Td1_attempted', 'Td2_attempted']:
# #         if col in round_df.columns:
# #             round_df[col] = round_df[col] / num_rounds
# #     round_df['Fight_Duration_Min'] = 5.0  # Each round is 5 minutes
# #     round_dfs.append(round_df)

# # # Concatenate round-wise data
# # df_rounds = pd.concat(round_dfs, ignore_index=True)

# # # ------------------ Feature Engineering ------------------
# # # 1. Strike Volume Rate (SVR)
# # df_rounds['SVR1'] = (df_rounds['SigStr1_landed'] + df_rounds['Head1_landed'] + df_rounds['Body1_landed'] + df_rounds['Leg1_landed']) / df_rounds['Fight_Duration_Min']
# # df_rounds['SVR2'] = (df_rounds['SigStr2_landed'] + df_rounds['Head2_landed'] + df_rounds['Body2_landed'] + df_rounds['Leg2_landed']) / df_rounds['Fight_Duration_Min']
# # df_rounds['SVR_Diff'] = df_rounds['SVR1'] - df_rounds['SVR2']

# # # 2. Defense Efficiency (DefEff)
# # df_rounds['DefEff1'] = 1 - (df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6))
# # df_rounds['DefEff2'] = 1 - (df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6))
# # df_rounds['DefEff_Diff'] = df_rounds['DefEff1'] - df_rounds['DefEff2']

# # # 3. Takedown Success Rate (TD_Success and TD_Diff)
# # epsilon = 1e-6
# # df_rounds['TD_Success1'] = df_rounds['Td1_landed'] / (df_rounds['Td1_attempted'] + epsilon)
# # df_rounds['TD_Success2'] = df_rounds['Td2_landed'] / (df_rounds['Td2_attempted'] + epsilon)
# # df_rounds['TD_Diff'] = df_rounds['TD_Success1'] - df_rounds['TD_Success2']

# # # 4. Output-Accuracy Tradeoff (OAT)
# # df_rounds['SLpM1'] = df_rounds['SigStr1_landed'] / df_rounds['Fight_Duration_Min']
# # df_rounds['SLpM2'] = df_rounds['SigStr2_landed'] / df_rounds['Fight_Duration_Min']
# # df_rounds['StrAcc1'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
# # df_rounds['StrAcc2'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
# # df_rounds['OAT1'] = df_rounds['SLpM1'] * df_rounds['StrAcc1']
# # df_rounds['OAT2'] = df_rounds['SLpM2'] * df_rounds['StrAcc2']
# # df_rounds['OAT_Diff'] = df_rounds['OAT1'] - df_rounds['OAT2']

# # # 5. Knockdown Rate (KDR)
# # df_rounds['KDR1'] = df_rounds['KD1'] / df_rounds['Fight_Duration_Min']
# # df_rounds['KDR2'] = df_rounds['KD2'] / df_rounds['Fight_Duration_Min']
# # df_rounds['KDR_Diff'] = df_rounds['KDR1'] - df_rounds['KDR2']

# # # 6. Submission Aggression (SubAgg)
# # df_rounds['SubAgg1'] = df_rounds['SubAtt1'] / df_rounds['Fight_Duration_Min']
# # df_rounds['SubAgg2'] = df_rounds['SubAtt2'] / df_rounds['Fight_Duration_Min']
# # df_rounds['SubAgg_Diff'] = df_rounds['SubAgg1'] - df_rounds['SubAgg2']

# # # 7. Experience Differential (Exp_Diff)
# # if 'Record1' in df_rounds.columns and 'Record2' in df_rounds.columns:
# #     def parse_record(record):
# #         if isinstance(record, str):
# #             wins, losses, draws = map(int, record.split('-'))
# #             return wins + losses + draws
# #         return 0
# #     df_rounds['TotalFights1'] = df_rounds['Record1'].apply(parse_record)
# #     df_rounds['TotalFights2'] = df_rounds['Record2'].apply(parse_record)
# #     df_rounds['Exp_Diff'] = df_rounds['TotalFights1'] - df_rounds['TotalFights2']
# # else:
# #     df_rounds['Exp_Diff'] = 0

# # # 8. Age Differential (Age_Diff)
# # if 'DOB1' in df_rounds.columns and 'DOB2' in df_rounds.columns and 'Event Date' in df_rounds.columns:
# #     df_rounds['DOB1'] = pd.to_datetime(df_rounds['DOB1'], errors='coerce')
# #     df_rounds['DOB2'] = pd.to_datetime(df_rounds['DOB2'], errors='coerce')
# #     df_rounds['Age1'] = (df_rounds['Event Date'] - df_rounds['DOB1']).dt.days / 365.25
# #     df_rounds['Age2'] = (df_rounds['Event Date'] - df_rounds['DOB2']).dt.days / 365.25
# #     df_rounds['Age_Diff'] = df_rounds['Age1'] - df_rounds['Age2']
# # else:
# #     df_rounds['Age_Diff'] = 0

# # # 9. Reach and Height Advantage (ReachAdv, HeightAdv)
# # if 'Reach1' in df_rounds.columns and 'Reach2' in df_rounds.columns:
# #     df_rounds['ReachAdv'] = df_rounds['Reach1'] - df_rounds['Reach2']
# # else:
# #     df_rounds['ReachAdv'] = 0
# # if 'Height1' in df_rounds.columns and 'Height2' in df_rounds.columns:
# #     df_rounds['HeightAdv'] = df_rounds['Height1'] - df_rounds['Height2']
# # else:
# #     df_rounds['HeightAdv'] = 0

# # # 10. Fight Control Index (FCI)
# # alpha = 1.0
# # df_rounds['FCI1'] = df_rounds['Ctrl1'] + alpha * df_rounds['Td1_landed']
# # df_rounds['FCI2'] = df_rounds['Ctrl2'] + alpha * df_rounds['Td2_landed']
# # df_rounds['FCI_Diff'] = df_rounds['FCI1'] - df_rounds['FCI2']

# # # Additional Features from Previous Code
# # df_rounds['SigStr1_efficiency'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
# # df_rounds['SigStr2_efficiency'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
# # df_rounds['Head1_efficiency'] = df_rounds['Head1_landed'] / (df_rounds['Head1_attempted'] + 1e-6)
# # df_rounds['Body1_efficiency'] = df_rounds['Body1_landed'] / (df_rounds['Body1_attempted'] + 1e-6)
# # df_rounds['Leg1_efficiency'] = df_rounds['Leg1_landed'] / (df_rounds['Leg1_attempted'] + 1e-6)
# # df_rounds['Head2_efficiency'] = df_rounds['Head2_landed'] / (df_rounds['Head2_attempted'] + 1e-6)
# # df_rounds['Body2_efficiency'] = df_rounds['Body2_landed'] / (df_rounds['Body2_attempted'] + 1e-6)
# # df_rounds['Leg2_efficiency'] = df_rounds['Leg2_landed'] / (df_rounds['Leg2_attempted'] + 1e-6)

# # # Define feature sets
# # features = [
# #     'SVR_Diff', 'DefEff_Diff', 'TD_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff',
# #     'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv', 'FCI_Diff',
# #     'SigStr1_efficiency', 'SigStr2_efficiency', 'Head1_efficiency', 'Body1_efficiency',
# #     'Leg1_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency'
# # ]
# # features_f1 = ['SVR1', 'DefEff1', 'OAT1', 'KDR1', 'SubAgg1', 'FCI1', 'TD_Success1',
# #                'SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']
# # features_f2 = ['SVR2', 'DefEff2', 'OAT2', 'KDR2', 'SubAgg2', 'FCI2', 'TD_Success2',
# #                'SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']

# # # Define target columns
# # target_cols = [
# #     'SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
# #     'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed'
# # ]

# # # Handle missing values
# # df_rounds.fillna(df_rounds.mean(numeric_only=True), inplace=True)

# # # Select two fighters for simulation
# # fighters = sorted(df['Fighter1'].unique())
# # selected_f1 = fighters[0]
# # selected_f2 = fighters[1] if len(fighters) > 1 else fighters[0]
# # print(f"Selected Fighters for Simulation: {selected_f1} vs {selected_f2}")

# # # Filter data for selected fighters
# # f1_df = df_rounds[(df_rounds['Fighter1'] == selected_f1) | (df_rounds['Fighter2'] == selected_f1)].copy()
# # f2_df = df_rounds[(df_rounds['Fighter1'] == selected_f2) | (df_rounds['Fighter2'] == selected_f2)].copy()

# # # Adjust features based on fighter position
# # def adjust_features(df, fighter, features_f1, features_f2):
# #     adjusted_df = pd.DataFrame(columns=features_f1)
# #     for idx, row in df.iterrows():
# #         if row['Fighter1'] == fighter:
# #             adjusted_row = row[features_f1]
# #         else:
# #             adjusted_row = row[features_f2]
# #             adjusted_row.index = features_f1
# #         adjusted_df = pd.concat([adjusted_df, adjusted_row.to_frame().T], ignore_index=True)
# #     return adjusted_df

# # f1_df_adjusted = adjust_features(f1_df, selected_f1, features_f1, features_f2) if not f1_df.empty else pd.DataFrame(columns=features_f1)
# # f2_df_adjusted = adjust_features(f2_df, selected_f2, features_f1, features_f2) if not f2_df.empty else pd.DataFrame(columns=features_f1)

# # # ------------------ Model Training ------------------
# # print("\nTraining Models...")
# # # Prepare data for training
# # train_df = df_rounds.dropna(subset=features + target_cols)
# # X = train_df[features]
# # y = train_df[target_cols]
# # rounds = train_df['Round']

# # # Feature Selection (across all rounds)
# # initial_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# # initial_model.fit(X, y['SigStr1_landed'])
# # feature_importance = pd.Series(initial_model.feature_importances_, index=features)
# # top_features = feature_importance.sort_values(ascending=False)[:10].index.tolist()
# # print(f"Selected Top Features: {top_features}")

# # # Train models for each round and target
# # models = {round_num: {} for round_num in range(1, num_rounds + 1)}
# # predictions = {round_num: {} for round_num in range(1, num_rounds + 1)}
# # train_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
# # test_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
# # param_grid = {
# #     'n_estimators': [100, 200],
# #     'learning_rate': [0.01, 0.05],
# #     'max_depth': [2, 3],
# #     'subsample': [0.8]
# # }

# # for round_num in range(1, num_rounds + 1):
# #     print(f"\nTraining models for Round {round_num}...")
# #     # Filter data for the current round
# #     round_mask = rounds == round_num
# #     X_round = X[round_mask][top_features]
# #     y_round = y[round_mask]
    
# #     # Split data
# #     X_train, X_test, y_train, y_test = train_test_split(X_round, y_round, test_size=0.2, random_state=42)
    
# #     # Scale features
# #     scaler = StandardScaler()
# #     X_train_scaled = scaler.fit_transform(X_train)
# #     X_test_scaled = scaler.transform(X_test)
    
# #     # Train models for each target
# #     for target in tqdm(target_cols, desc=f"Training Round {round_num} models"):
# #         gb = GradientBoostingRegressor(
# #             random_state=42,
# #             n_iter_no_change=10,
# #             validation_fraction=0.1,
# #             tol=1e-4
# #         )
# #         grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
# #         grid_search.fit(X_train_scaled, y_train[target])
# #         models[round_num][target] = grid_search.best_estimator_
# #         predictions[round_num][target] = grid_search.best_estimator_.predict(X_test_scaled)
# #         # Compute training and test R²
# #         train_r2[round_num][target] = r2_score(y_train[target], grid_search.best_estimator_.predict(X_train_scaled))
# #         test_r2[round_num][target] = r2_score(y_test[target], predictions[round_num][target])
# #         print(f"Best Parameters for {target} (Round {round_num}): {grid_search.best_params_}")

# #     # ------------------ Evaluation Metrics ------------------
# #     print(f"\nModel Performance for Round {round_num} (Training vs. Test):")
# #     for target in target_cols:
# #         train_mae = mean_absolute_error(y_train[target], models[round_num][target].predict(X_train_scaled))
# #         test_mae = mean_absolute_error(y_test[target], predictions[round_num][target])
# #         train_mse = mean_squared_error(y_train[target], models[round_num][target].predict(X_train_scaled))
# #         test_mse = mean_squared_error(y_test[target], predictions[round_num][target])
# #         print(f"{target} Prediction:")
# #         print(f"Training MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
# #         print(f"Training MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
# #         print(f"Training Accuracy (R²): {train_r2[round_num][target]:.2f}, Test Accuracy (R²): {test_r2[round_num][target]:.2f}")
# #         print(f"Overfitting Indicator (Train-Test R² Gap): {(train_r2[round_num][target] - test_r2[round_num][target]):.2f}")

# # # ------------------ Fight Simulation ------------------
# # print(f"\nFight Simulation: {selected_f1} vs {selected_f2}")
# # if not f1_df_adjusted.empty and not f2_df_adjusted.empty:
# #     f1_input = f1_df_adjusted.mean()
# #     f2_input = f2_df_adjusted.mean()
# #     input_diff = pd.Series(index=top_features)
# #     for f1, diff in zip(features_f1[:6], ['SVR_Diff', 'DefEff_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff', 'FCI_Diff']):
# #         if diff in top_features:
# #             input_diff[diff] = f1_input[f1] - f2_input[f1]
# #     for feature in ['TD_Diff', 'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv']:
# #         if feature in top_features:
# #             input_diff[feature] = f1_df[feature].mean() if feature in f1_df.columns else 0
# #     for feature in ['SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']:
# #         if feature in top_features:
# #             input_diff[feature] = f1_input[feature] if feature in f1_input.index else 0
# #     for feature in ['SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']:
# #         if feature in top_features:
# #             input_diff[feature] = f2_input[feature.replace('2', '1')] if feature.replace('2', '1') in f2_input.index else 0

# #     input_row = input_diff.values.reshape(1, -1)
# #     input_scaled = scaler.transform(input_row)

# #     total_predictions = {target: 0 for target in target_cols}
# #     for round_num in range(1, num_rounds + 1):
# #         print(f"\nRound {round_num} Predictions:")
# #         for target in target_cols:
# #             pred = models[round_num][target].predict(input_scaled)[0]
# #             total_predictions[target] += pred
# #             print(f"{target}: {pred:.2f}")

# #     print(f"\nTotal Fight Predictions:")
# #     for target in target_cols:
# #         print(f"{target}: {total_predictions[target]:.2f}")
# # else:
# #     print("Insufficient data for simulation.")

# # # ------------------ Plots ------------------
# # # Use Round 1 data for plots
# # round_mask = rounds == 1
# # X_round1 = X[round_mask][top_features]
# # y_round1 = y[round_mask]
# # X_train, X_test, y_train, y_test = train_test_split(X_round1, y_round1, test_size=0.2, random_state=42)
# # X_train_scaled = scaler.fit_transform(X_train)
# # X_test_scaled = scaler.transform(X_test)

# # # Learning Curves (for Round 1)
# # def plot_learning_curve(estimator, X, y, title, filename):
# #     train_sizes, train_scores, test_scores = learning_curve(
# #         estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
# #         scoring='neg_mean_absolute_error')
# #     train_scores_mean = -train_scores.mean(axis=1)
# #     test_scores_mean = -test_scores.mean(axis=1)
    
# #     plt.figure(figsize=(10, 6))
# #     plt.plot(train_sizes, train_scores_mean, label='Training MAE')
# #     plt.plot(train_sizes, test_scores_mean, label='Validation MAE')
# #     plt.title(title)
# #     plt.xlabel('Training Examples')
# #     plt.ylabel('Mean Absolute Error')
# #     plt.legend(loc='best')
# #     plt.grid(True)
# #     plt.savefig(filename)
# #     plt.show()  # Display the plot (removed plt.close() to keep it open)

# # for target in target_cols:
# #     plot_learning_curve(
# #         models[1][target], 
# #         X_train_scaled, 
# #         y_train[target], 
# #         f'Learning Curve {target} (Round 1)', 
# #         f'learning_curve_round1_{target}.png'
# #     )

# # # Predicted vs Actual (Round 1)
# # plt.figure(figsize=(10, 6))
# # for target in target_cols:
# #     plt.scatter(y_test[target], predictions[1][target], alpha=0.5, label=target)
# # plt.plot([y_test.min().min(), y_test.max().max()], [y_test.min().min(), y_test.max().max()], 'r--')
# # plt.xlabel('Actual Strikes')
# # plt.ylabel('Predicted Strikes')
# # plt.title('Predicted vs Actual Strikes (Round 1)')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('predicted_vs_actual_round1.png')
# # plt.show()  # Added to display the plot (removed plt.close())

# # # Residual Plot (Round 1)
# # plt.figure(figsize=(10, 6))
# # for target in target_cols:
# #     residuals = y_test[target] - predictions[1][target]
# #     sns.histplot(residuals, kde=True, label=target, alpha=0.5)
# # plt.title('Residual Distribution (Round 1)')
# # plt.xlabel('Residuals')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig('residual_distribution_round1.png')
# # plt.show()  # Added to display the plot (removed plt.close())

# # # Feature Importance (Round 1)
# # plt.figure(figsize=(12, 8))
# # for target in target_cols:
# #     feature_importance = pd.Series(models[1][target].feature_importances_, index=top_features)
# #     feature_importance.sort_values(ascending=False)[:10].plot(kind='bar', alpha=0.5, label=target)
# # plt.title('Top 10 Feature Importances (Round 1)')
# # plt.xlabel('Features')
# # plt.ylabel('Importance')
# # plt.legend()
# # plt.xticks(rotation=45)
# # plt.tight_layout()
# # plt.savefig('feature_importance_round1.png')
# # plt.show()  # Added to display the plot (removed plt.close())





# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# import warnings

# # Suppress warnings for cleaner output
# warnings.filterwarnings('ignore')

# # ------------------ Load Data ------------------
# def load_data():
#     df = pd.read_csv("C:/Users/abhi0/OneDrive/Desktop/IIITH/SMAI/PROJECT_clean_dataset/preprocessed_fight_fighters_data_dates_formatted.csv")
#     df = df.dropna(subset=["winner", "SigStr1_landed", "Ctrl1", "KD1", "Td1_landed"])
#     df["Win"] = (df['winner'].str.lower() == df['Fighter1'].str.lower()).astype(int)
#     return df

# df = load_data()

# # Add Fight_Duration_Min if not present (assuming 15 minutes for 3-round fights)
# if 'Fight_Duration_Min' not in df.columns:
#     df['Fight_Duration_Min'] = 15.0  # Placeholder; adjust based on actual data

# # Check for event date column and standardize its name
# possible_date_columns = ['Event Date', 'event_date', 'Date', 'date', 'event date']
# event_date_col = None
# for col in possible_date_columns:
#     if col in df.columns:
#         event_date_col = col
#         break

# # Flag to track if dates are synthetic
# is_synthetic_dates = False

# if event_date_col:
#     df.rename(columns={event_date_col: 'Event Date'}, inplace=True)
# else:
#     # Create synthetic Event Date
#     is_synthetic_dates = True
#     start_date = pd.Timestamp('2020-01-01')
#     end_date = pd.Timestamp('2024-12-31')
#     max_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
#     num_rows = len(df)
#     if num_rows <= max_months:
#         dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=num_rows)
#     else:
#         dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=max_months)
#         dates = np.tile(dates, (num_rows // max_months) + 1)[:num_rows]
#     df['Event Date'] = dates

# # Convert numeric Event Date to datetime
# if not is_synthetic_dates:
#     df['Event Date'] = df['Event Date'].astype(str)
#     df['Event Date'] = pd.to_datetime(df['Event Date'], format='%Y%m%d', errors='coerce')

# # Warn about NaN dates
# nan_dates = df['Event Date'].isna().sum()
# if nan_dates > 0:
#     print(f"Warning: Found {nan_dates} rows with invalid dates in 'Event Date'. These will be excluded from the timeline.")

# # ------------------ Create Round-Wise Data ------------------
# # Assume 3 rounds per fight, divide fight-level metrics by 3
# num_rounds = 3
# round_dfs = []
# for round_num in range(1, num_rounds + 1):
#     round_df = df.copy()
#     round_df['Round'] = round_num
#     # Divide fight-level metrics by number of rounds
#     for col in ['SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
#                 'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed',
#                 'Ctrl1', 'Ctrl2', 'KD1', 'KD2', 'Td1_landed', 'Td2_landed',
#                 'SubAtt1', 'SubAtt2', 'SigStr1_attempted', 'SigStr2_attempted',
#                 'Head1_attempted', 'Head2_attempted', 'Body1_attempted', 'Body2_attempted',
#                 'Leg1_attempted', 'Leg2_attempted', 'Td1_attempted', 'Td2_attempted']:
#         if col in round_df.columns:
#             round_df[col] = round_df[col] / num_rounds
#     round_df['Fight_Duration_Min'] = 5.0  # Each round is 5 minutes
#     round_dfs.append(round_df)

# # Concatenate round-wise data
# df_rounds = pd.concat(round_dfs, ignore_index=True)

# # ------------------ Feature Engineering ------------------
# # 1. Strike Volume Rate (SVR)
# df_rounds['SVR1'] = (df_rounds['SigStr1_landed'] + df_rounds['Head1_landed'] + df_rounds['Body1_landed'] + df_rounds['Leg1_landed']) / df_rounds['Fight_Duration_Min']
# df_rounds['SVR2'] = (df_rounds['SigStr2_landed'] + df_rounds['Head2_landed'] + df_rounds['Body2_landed'] + df_rounds['Leg2_landed']) / df_rounds['Fight_Duration_Min']
# df_rounds['SVR_Diff'] = df_rounds['SVR1'] - df_rounds['SVR2']

# # 2. Defense Efficiency (DefEff)
# df_rounds['DefEff1'] = 1 - (df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6))
# df_rounds['DefEff2'] = 1 - (df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6))
# df_rounds['DefEff_Diff'] = df_rounds['DefEff1'] - df_rounds['DefEff2']

# # 3. Takedown Success Rate (TD_Success and TD_Diff)
# epsilon = 1e-6
# df_rounds['TD_Success1'] = df_rounds['Td1_landed'] / (df_rounds['Td1_attempted'] + epsilon)
# df_rounds['TD_Success2'] = df_rounds['Td2_landed'] / (df_rounds['Td2_attempted'] + epsilon)
# df_rounds['TD_Diff'] = df_rounds['TD_Success1'] - df_rounds['TD_Success2']

# # 4. Output-Accuracy Tradeoff (OAT)
# df_rounds['SLpM1'] = df_rounds['SigStr1_landed'] / df_rounds['Fight_Duration_Min']
# df_rounds['SLpM2'] = df_rounds['SigStr2_landed'] / df_rounds['Fight_Duration_Min']
# df_rounds['StrAcc1'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
# df_rounds['StrAcc2'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
# df_rounds['OAT1'] = df_rounds['SLpM1'] * df_rounds['StrAcc1']
# df_rounds['OAT2'] = df_rounds['SLpM2'] * df_rounds['StrAcc2']
# df_rounds['OAT_Diff'] = df_rounds['OAT1'] - df_rounds['OAT2']

# # 5. Knockdown Rate (KDR)
# df_rounds['KDR1'] = df_rounds['KD1'] / df_rounds['Fight_Duration_Min']
# df_rounds['KDR2'] = df_rounds['KD2'] / df_rounds['Fight_Duration_Min']
# df_rounds['KDR_Diff'] = df_rounds['KDR1'] - df_rounds['KDR2']

# # 6. Submission Aggression (SubAgg)
# df_rounds['SubAgg1'] = df_rounds['SubAtt1'] / df_rounds['Fight_Duration_Min']
# df_rounds['SubAgg2'] = df_rounds['SubAtt2'] / df_rounds['Fight_Duration_Min']
# df_rounds['SubAgg_Diff'] = df_rounds['SubAgg1'] - df_rounds['SubAgg2']

# # 7. Experience Differential (Exp_Diff)
# if 'Record1' in df_rounds.columns and 'Record2' in df_rounds.columns:
#     def parse_record(record):
#         if isinstance(record, str):
#             wins, losses, draws = map(int, record.split('-'))
#             return wins + losses + draws
#         return 0
#     df_rounds['TotalFights1'] = df_rounds['Record1'].apply(parse_record)
#     df_rounds['TotalFights2'] = df_rounds['Record2'].apply(parse_record)
#     df_rounds['Exp_Diff'] = df_rounds['TotalFights1'] - df_rounds['TotalFights2']
# else:
#     df_rounds['Exp_Diff'] = 0

# # 8. Age Differential (Age_Diff)
# if 'DOB1' in df_rounds.columns and 'DOB2' in df_rounds.columns and 'Event Date' in df_rounds.columns:
#     df_rounds['DOB1'] = pd.to_datetime(df_rounds['DOB1'], errors='coerce')
#     df_rounds['DOB2'] = pd.to_datetime(df_rounds['DOB2'], errors='coerce')
#     df_rounds['Age1'] = (df_rounds['Event Date'] - df_rounds['DOB1']).dt.days / 365.25
#     df_rounds['Age2'] = (df_rounds['Event Date'] - df_rounds['DOB2']).dt.days / 365.25
#     df_rounds['Age_Diff'] = df_rounds['Age1'] - df_rounds['Age2']
# else:
#     df_rounds['Age_Diff'] = 0

# # 9. Reach and Height Advantage (ReachAdv, HeightAdv)
# if 'Reach1' in df_rounds.columns and 'Reach2' in df_rounds.columns:
#     df_rounds['ReachAdv'] = df_rounds['Reach1'] - df_rounds['Reach2']
# else:
#     df_rounds['ReachAdv'] = 0
# if 'Height1' in df_rounds.columns and 'Height2' in df_rounds.columns:
#     df_rounds['HeightAdv'] = df_rounds['Height1'] - df_rounds['Height2']
# else:
#     df_rounds['HeightAdv'] = 0

# # 10. Fight Control Index (FCI)
# alpha = 1.0
# df_rounds['FCI1'] = df_rounds['Ctrl1'] + alpha * df_rounds['Td1_landed']
# df_rounds['FCI2'] = df_rounds['Ctrl2'] + alpha * df_rounds['Td2_landed']
# df_rounds['FCI_Diff'] = df_rounds['FCI1'] - df_rounds['FCI2']

# # Additional Features from Previous Code
# df_rounds['SigStr1_efficiency'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
# df_rounds['SigStr2_efficiency'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
# df_rounds['Head1_efficiency'] = df_rounds['Head1_landed'] / (df_rounds['Head1_attempted'] + 1e-6)
# df_rounds['Body1_efficiency'] = df_rounds['Body1_landed'] / (df_rounds['Body1_attempted'] + 1e-6)
# df_rounds['Leg1_efficiency'] = df_rounds['Leg1_landed'] / (df_rounds['Leg1_attempted'] + 1e-6)
# df_rounds['Head2_efficiency'] = df_rounds['Head2_landed'] / (df_rounds['Head2_attempted'] + 1e-6)
# df_rounds['Body2_efficiency'] = df_rounds['Body2_landed'] / (df_rounds['Body2_attempted'] + 1e-6)
# df_rounds['Leg2_efficiency'] = df_rounds['Leg2_landed'] / (df_rounds['Leg2_attempted'] + 1e-6)

# # Define feature sets
# features = [
#     'SVR_Diff', 'DefEff_Diff', 'TD_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff',
#     'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv', 'FCI_Diff',
#     'SigStr1_efficiency', 'SigStr2_efficiency', 'Head1_efficiency', 'Body1_efficiency',
#     'Leg1_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency'
# ]
# features_f1 = ['SVR1', 'DefEff1', 'OAT1', 'KDR1', 'SubAgg1', 'FCI1', 'TD_Success1',
#                'SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']
# features_f2 = ['SVR2', 'DefEff2', 'OAT2', 'KDR2', 'SubAgg2', 'FCI2', 'TD_Success2',
#                'SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']

# # Define target columns
# target_cols = [
#     'SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
#     'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed'
# ]

# # Handle missing values
# df_rounds.fillna(df_rounds.mean(numeric_only=True), inplace=True)

# # Select two fighters for simulation
# fighters = sorted(df['Fighter1'].unique())
# selected_f1 = fighters[0]
# selected_f2 = fighters[1] if len(fighters) > 1 else fighters[0]
# print(f"Selected Fighters for Simulation: {selected_f1} vs {selected_f2}")

# # Filter data for selected fighters
# f1_df = df_rounds[(df_rounds['Fighter1'] == selected_f1) | (df_rounds['Fighter2'] == selected_f1)].copy()
# f2_df = df_rounds[(df_rounds['Fighter1'] == selected_f2) | (df_rounds['Fighter2'] == selected_f2)].copy()

# # Adjust features based on fighter position
# def adjust_features(df, fighter, features_f1, features_f2):
#     adjusted_df = pd.DataFrame(columns=features_f1)
#     for idx, row in df.iterrows():
#         if row['Fighter1'] == fighter:
#             adjusted_row = row[features_f1]
#         else:
#             adjusted_row = row[features_f2]
#             adjusted_row.index = features_f1
#         adjusted_df = pd.concat([adjusted_df, adjusted_row.to_frame().T], ignore_index=True)
#     return adjusted_df

# f1_df_adjusted = adjust_features(f1_df, selected_f1, features_f1, features_f2) if not f1_df.empty else pd.DataFrame(columns=features_f1)
# f2_df_adjusted = adjust_features(f2_df, selected_f2, features_f1, features_f2) if not f2_df.empty else pd.DataFrame(columns=features_f1)

# # ------------------ Model Training ------------------
# print("\nTraining Models...")
# # Prepare data for training
# train_df = df_rounds.dropna(subset=features + target_cols)
# X = train_df[features]
# y = train_df[target_cols]
# rounds = train_df['Round']

# # Feature Selection (across all rounds)
# initial_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# initial_model.fit(X, y['SigStr1_landed'])
# feature_importance = pd.Series(initial_model.feature_importances_, index=features)
# top_features = feature_importance.sort_values(ascending=False)[:10].index.tolist()
# print(f"Selected Top Features: {top_features}")

# # Train models for each round and target
# models = {round_num: {} for round_num in range(1, num_rounds + 1)}
# predictions = {round_num: {} for round_num in range(1, num_rounds + 1)}
# train_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
# test_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
# param_grid = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.01, 0.05],
#     'max_depth': [2, 3],
#     'subsample': [0.8]
# }

# for round_num in range(1, num_rounds + 1):
#     print(f"\nTraining models for Round {round_num}...")
#     # Filter data for the current round
#     round_mask = rounds == round_num
#     X_round = X[round_mask][top_features]
#     y_round = y[round_mask]
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X_round, y_round, test_size=0.2, random_state=42)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train models for each target
#     for target in tqdm(target_cols, desc=f"Training Round {round_num} models"):
#         gb = GradientBoostingRegressor(
#             random_state=42,
#             n_iter_no_change=10,
#             validation_fraction=0.1,
#             tol=1e-4
#         )
#         grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
#         grid_search.fit(X_train_scaled, y_train[target])
#         models[round_num][target] = grid_search.best_estimator_
#         predictions[round_num][target] = grid_search.best_estimator_.predict(X_test_scaled)
#         # Compute training and test R²
#         train_r2[round_num][target] = r2_score(y_train[target], grid_search.best_estimator_.predict(X_train_scaled))
#         test_r2[round_num][target] = r2_score(y_test[target], predictions[round_num][target])
#         print(f"Best Parameters for {target} (Round {round_num}): {grid_search.best_params_}")

#     # ------------------ Evaluation Metrics ------------------
#     print(f"\nModel Performance for Round {round_num} (Training vs. Test):")
#     for target in target_cols:
#         train_mae = mean_absolute_error(y_train[target], models[round_num][target].predict(X_train_scaled))
#         test_mae = mean_absolute_error(y_test[target], predictions[round_num][target])
#         train_mse = mean_squared_error(y_train[target], models[round_num][target].predict(X_train_scaled))
#         test_mse = mean_squared_error(y_test[target], predictions[round_num][target])
#         print(f"{target} Prediction:")
#         print(f"Training MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
#         print(f"Training MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
#         print(f"Training Accuracy (R²): {train_r2[round_num][target]:.2f}, Test Accuracy (R²): {test_r2[round_num][target]:.2f}")
#         print(f"Overfitting Indicator (Train-Test R² Gap): {(train_r2[round_num][target] - test_r2[round_num][target]):.2f}")

# # ------------------ Fight Simulation ------------------
# print(f"\nFight Simulation: {selected_f1} vs {selected_f2}")
# if not f1_df_adjusted.empty and not f2_df_adjusted.empty:
#     f1_input = f1_df_adjusted.mean()
#     f2_input = f2_df_adjusted.mean()
#     input_diff = pd.Series(index=top_features)
#     for f1, diff in zip(features_f1[:6], ['SVR_Diff', 'DefEff_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff', 'FCI_Diff']):
#         if diff in top_features:
#             input_diff[diff] = f1_input[f1] - f2_input[f1]
#     for feature in ['TD_Diff', 'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv']:
#         if feature in top_features:
#             input_diff[feature] = f1_df[feature].mean() if feature in f1_df.columns else 0
#     for feature in ['SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']:
#         if feature in top_features:
#             input_diff[feature] = f1_input[feature] if feature in f1_input.index else 0
#     for feature in ['SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']:
#         if feature in top_features:
#             input_diff[feature] = f2_input[feature.replace('2', '1')] if feature.replace('2', '1') in f2_input.index else 0

#     input_row = input_diff.values.reshape(1, -1)
#     input_scaled = scaler.transform(input_row)

#     total_predictions = {target: 0 for target in target_cols}
#     for round_num in range(1, num_rounds + 1):
#         print(f"\nRound {round_num} Predictions:")
#         for target in target_cols:
#             pred = models[round_num][target].predict(input_scaled)[0]
#             total_predictions[target] += pred
#             print(f"{target}: {pred:.2f}")

#     print(f"\nTotal Fight Predictions:")
#     for target in target_cols:
#         print(f"{target}: {total_predictions[target]:.2f}")
# else:
#     print("Insufficient data for simulation.")

# # ------------------ Plots ------------------
# # Use Round 1 data for plots
# round_mask = rounds == 1
# X_round1 = X[round_mask][top_features]
# y_round1 = y[round_mask]
# X_train, X_test, y_train, y_test = train_test_split(X_round1, y_round1, test_size=0.2, random_state=42)
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Learning Curves (for Round 1)
# def plot_learning_curve(estimator, X, y, title, filename):
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
#         scoring='neg_mean_squared_error')
#     train_scores_mean = -train_scores.mean(axis=1)
#     test_scores_mean = -test_scores.mean(axis=1)
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_sizes, train_scores_mean, label='Training MSE')
#     plt.plot(train_sizes, test_scores_mean, label='Validation MSE')
#     plt.title(title)
#     plt.xlabel('Training Examples')
#     plt.ylabel('Mean Squared Error')
#     plt.legend(loc='best')
#     plt.grid(True)
#     plt.savefig(filename)
#     plt.show()

# for target in target_cols:
#     plot_learning_curve(
#         models[1][target], 
#         X_train_scaled, 
#         y_train[target], 
#         f'Learning Curve {target} (Round 1)', 
#         f'learning_curve_round1_{target}.png'
#     )

# # Predicted vs Actual (Round 1)
# plt.figure(figsize=(10, 6))
# for target in target_cols:
#     plt.scatter(y_test[target], predictions[1][target], alpha=0.5, label=target)
# plt.plot([y_test.min().min(), y_test.max().max()], [y_test.min().min(), y_test.max().max()], 'r--')
# plt.xlabel('Actual Strikes')
# plt.ylabel('Predicted Strikes')
# plt.title('Predicted vs Actual Strikes (Round 1)')
# plt.legend()
# plt.grid(True)
# plt.savefig('predicted_vs_actual_round1.png')
# plt.show()

# # Residual Plot (Round 1)
# plt.figure(figsize=(10, 6))
# for target in target_cols:
#     residuals = y_test[target] - predictions[1][target]
#     sns.histplot(residuals, kde=True, label=target, alpha=0.5)
# plt.title('Residual Distribution (Round 1)')
# plt.xlabel('Residuals')
# plt.legend()
# plt.grid(True)
# plt.savefig('residual_distribution_round1.png')
# plt.show()

# # Feature Importance (Round 1)
# plt.figure(figsize=(12, 8))
# for target in target_cols:
#     feature_importance = pd.Series(models[1][target].feature_importances_, index=top_features)
#     feature_importance.sort_values(ascending=False)[:10].plot(kind='bar', alpha=0.5, label=target)
# plt.title('Top 10 Feature Importances (Round 1)')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('feature_importance_round1.png')
# plt.show()



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ------------------ Load Data ------------------
def load_data():
    df = pd.read_csv("C:/Users/abhi0/OneDrive/Desktop/IIITH/SMAI/PROJECT_clean_dataset/preprocessed_fight_fighters_data_dates_formatted.csv")
    df = df.dropna(subset=["winner", "SigStr1_landed", "Ctrl1", "KD1", "Td1_landed"])
    df["Win"] = (df['winner'].str.lower() == df['Fighter1'].str.lower()).astype(int)
    return df

df = load_data()

# Add Fight_Duration_Min if not present (assuming 15 minutes for 3-round fights)
if 'Fight_Duration_Min' not in df.columns:
    df['Fight_Duration_Min'] = 15.0  # Placeholder; adjust based on actual data

# Check for event date column and standardize its name
possible_date_columns = ['Event Date', 'event_date', 'Date', 'date', 'event date']
event_date_col = None
for col in possible_date_columns:
    if col in df.columns:
        event_date_col = col
        break

# Flag to track if dates are synthetic
is_synthetic_dates = False

if event_date_col:
    df.rename(columns={event_date_col: 'Event Date'}, inplace=True)
else:
    # Create synthetic Event Date
    is_synthetic_dates = True
    start_date = pd.Timestamp('2020-01-01')
    end_date = pd.Timestamp('2024-12-31')
    max_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    num_rows = len(df)
    if num_rows <= max_months:
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=num_rows)
    else:
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=max_months)
        dates = np.tile(dates, (num_rows // max_months) + 1)[:num_rows]
    df['Event Date'] = dates

# Convert numeric Event Date to datetime
if not is_synthetic_dates:
    df['Event Date'] = df['Event Date'].astype(str)
    df['Event Date'] = pd.to_datetime(df['Event Date'], format='%Y%m%d', errors='coerce')

# Warn about NaN dates
nan_dates = df['Event Date'].isna().sum()
if nan_dates > 0:
    print(f"Warning: Found {nan_dates} rows with invalid dates in 'Event Date'. These will be excluded from the timeline.")

# ------------------ Create Round-Wise Data ------------------
# Assume 3 rounds per fight, divide fight-level metrics by 3
num_rounds = 3
round_dfs = []
for round_num in range(1, num_rounds + 1):
    round_df = df.copy()
    round_df['Round'] = round_num
    # Divide fight-level metrics by number of rounds
    for col in ['SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
                'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed',
                'Ctrl1', 'Ctrl2', 'KD1', 'KD2', 'Td1_landed', 'Td2_landed',
                'SubAtt1', 'SubAtt2', 'SigStr1_attempted', 'SigStr2_attempted',
                'Head1_attempted', 'Head2_attempted', 'Body1_attempted', 'Body2_attempted',
                'Leg1_attempted', 'Leg2_attempted', 'Td1_attempted', 'Td2_attempted']:
        if col in round_df.columns:
            round_df[col] = round_df[col] / num_rounds
    round_df['Fight_Duration_Min'] = 5.0  # Each round is 5 minutes
    round_dfs.append(round_df)

# Concatenate round-wise data
df_rounds = pd.concat(round_dfs, ignore_index=True)

# ------------------ Feature Engineering ------------------
# 1. Strike Volume Rate (SVR)
df_rounds['SVR1'] = (df_rounds['SigStr1_landed'] + df_rounds['Head1_landed'] + df_rounds['Body1_landed'] + df_rounds['Leg1_landed']) / df_rounds['Fight_Duration_Min']
df_rounds['SVR2'] = (df_rounds['SigStr2_landed'] + df_rounds['Head2_landed'] + df_rounds['Body2_landed'] + df_rounds['Leg2_landed']) / df_rounds['Fight_Duration_Min']
df_rounds['SVR_Diff'] = df_rounds['SVR1'] - df_rounds['SVR2']

# 2. Defense Efficiency (DefEff)
df_rounds['DefEff1'] = 1 - (df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6))
df_rounds['DefEff2'] = 1 - (df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6))
df_rounds['DefEff_Diff'] = df_rounds['DefEff1'] - df_rounds['DefEff2']

# 3. Takedown Success Rate (TD_Success and TD_Diff)
epsilon = 1e-6
df_rounds['TD_Success1'] = df_rounds['Td1_landed'] / (df_rounds['Td1_attempted'] + epsilon)
df_rounds['TD_Success2'] = df_rounds['Td2_landed'] / (df_rounds['Td2_attempted'] + epsilon)
df_rounds['TD_Diff'] = df_rounds['TD_Success1'] - df_rounds['TD_Success2']

# 4. Output-Accuracy Tradeoff (OAT)
df_rounds['SLpM1'] = df_rounds['SigStr1_landed'] / df_rounds['Fight_Duration_Min']
df_rounds['SLpM2'] = df_rounds['SigStr2_landed'] / df_rounds['Fight_Duration_Min']
df_rounds['StrAcc1'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
df_rounds['StrAcc2'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
df_rounds['OAT1'] = df_rounds['SLpM1'] * df_rounds['StrAcc1']
df_rounds['OAT2'] = df_rounds['SLpM2'] * df_rounds['StrAcc2']
df_rounds['OAT_Diff'] = df_rounds['OAT1'] - df_rounds['OAT2']

# 5. Knockdown Rate (KDR)
df_rounds['KDR1'] = df_rounds['KD1'] / df_rounds['Fight_Duration_Min']
df_rounds['KDR2'] = df_rounds['KD2'] / df_rounds['Fight_Duration_Min']
df_rounds['KDR_Diff'] = df_rounds['KDR1'] - df_rounds['KDR2']

# 6. Submission Aggression (SubAgg)
df_rounds['SubAgg1'] = df_rounds['SubAtt1'] / df_rounds['Fight_Duration_Min']
df_rounds['SubAgg2'] = df_rounds['SubAtt2'] / df_rounds['Fight_Duration_Min']
df_rounds['SubAgg_Diff'] = df_rounds['SubAgg1'] - df_rounds['SubAgg2']

# 7. Experience Differential (Exp_Diff)
if 'Record1' in df_rounds.columns and 'Record2' in df_rounds.columns:
    def parse_record(record):
        if isinstance(record, str):
            wins, losses, draws = map(int, record.split('-'))
            return wins + losses + draws
        return 0
    df_rounds['TotalFights1'] = df_rounds['Record1'].apply(parse_record)
    df_rounds['TotalFights2'] = df_rounds['Record2'].apply(parse_record)
    df_rounds['Exp_Diff'] = df_rounds['TotalFights1'] - df_rounds['TotalFights2']
else:
    df_rounds['Exp_Diff'] = 0

# 8. Age Differential (Age_Diff)
if 'DOB1' in df_rounds.columns and 'DOB2' in df_rounds.columns and 'Event Date' in df_rounds.columns:
    df_rounds['DOB1'] = pd.to_datetime(df_rounds['DOB1'], errors='coerce')
    df_rounds['DOB2'] = pd.to_datetime(df_rounds['DOB2'], errors='coerce')
    df_rounds['Age1'] = (df_rounds['Event Date'] - df_rounds['DOB1']).dt.days / 365.25
    df_rounds['Age2'] = (df_rounds['Event Date'] - df_rounds['DOB2']).dt.days / 365.25
    df_rounds['Age_Diff'] = df_rounds['Age1'] - df_rounds['Age2']
else:
    df_rounds['Age_Diff'] = 0

# 9. Reach and Height Advantage (ReachAdv, HeightAdv)
if 'Reach1' in df_rounds.columns and 'Reach2' in df_rounds.columns:
    df_rounds['ReachAdv'] = df_rounds['Reach1'] - df_rounds['Reach2']
else:
    df_rounds['ReachAdv'] = 0
if 'Height1' in df_rounds.columns and 'Height2' in df_rounds.columns:
    df_rounds['HeightAdv'] = df_rounds['Height1'] - df_rounds['Height2']
else:
    df_rounds['HeightAdv'] = 0

# 10. Fight Control Index (FCI)
alpha = 1.0
df_rounds['FCI1'] = df_rounds['Ctrl1'] + alpha * df_rounds['Td1_landed']
df_rounds['FCI2'] = df_rounds['Ctrl2'] + alpha * df_rounds['Td2_landed']
df_rounds['FCI_Diff'] = df_rounds['FCI1'] - df_rounds['FCI2']

# Additional Features from Previous Code
df_rounds['SigStr1_efficiency'] = df_rounds['SigStr1_landed'] / (df_rounds['SigStr1_attempted'] + 1e-6)
df_rounds['SigStr2_efficiency'] = df_rounds['SigStr2_landed'] / (df_rounds['SigStr2_attempted'] + 1e-6)
df_rounds['Head1_efficiency'] = df_rounds['Head1_landed'] / (df_rounds['Head1_attempted'] + 1e-6)
df_rounds['Body1_efficiency'] = df_rounds['Body1_landed'] / (df_rounds['Body1_attempted'] + 1e-6)
df_rounds['Leg1_efficiency'] = df_rounds['Leg1_landed'] / (df_rounds['Leg1_attempted'] + 1e-6)
df_rounds['Head2_efficiency'] = df_rounds['Head2_landed'] / (df_rounds['Head2_attempted'] + 1e-6)
df_rounds['Body2_efficiency'] = df_rounds['Body2_landed'] / (df_rounds['Body2_attempted'] + 1e-6)
df_rounds['Leg2_efficiency'] = df_rounds['Leg2_landed'] / (df_rounds['Leg2_attempted'] + 1e-6)

# Define feature sets
features = [
    'SVR_Diff', 'DefEff_Diff', 'TD_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff',
    'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv', 'FCI_Diff',
    'SigStr1_efficiency', 'SigStr2_efficiency', 'Head1_efficiency', 'Body1_efficiency',
    'Leg1_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency'
]
features_f1 = ['SVR1', 'DefEff1', 'OAT1', 'KDR1', 'SubAgg1', 'FCI1', 'TD_Success1',
               'SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']
features_f2 = ['SVR2', 'DefEff2', 'OAT2', 'KDR2', 'SubAgg2', 'FCI2', 'TD_Success2',
               'SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']

# Define target columns
target_cols = [
    'SigStr1_landed', 'Head1_landed', 'Body1_landed', 'Leg1_landed',
    'SigStr2_landed', 'Head2_landed', 'Body2_landed', 'Leg2_landed'
]

# Handle missing values
df_rounds.fillna(df_rounds.mean(numeric_only=True), inplace=True)

# Select two fighters for simulation
fighters = sorted(df['Fighter1'].unique())
selected_f1 = fighters[0]
selected_f2 = fighters[1] if len(fighters) > 1 else fighters[0]
print(f"Selected Fighters for Simulation: {selected_f1} vs {selected_f2}")

# Filter data for selected fighters
f1_df = df_rounds[(df_rounds['Fighter1'] == selected_f1) | (df_rounds['Fighter2'] == selected_f1)].copy()
f2_df = df_rounds[(df_rounds['Fighter1'] == selected_f2) | (df_rounds['Fighter2'] == selected_f2)].copy()

# Adjust features based on fighter position
def adjust_features(df, fighter, features_f1, features_f2):
    adjusted_df = pd.DataFrame(columns=features_f1)
    for idx, row in df.iterrows():
        if row['Fighter1'] == fighter:
            adjusted_row = row[features_f1]
        else:
            adjusted_row = row[features_f2]
            adjusted_row.index = features_f1
        adjusted_df = pd.concat([adjusted_df, adjusted_row.to_frame().T], ignore_index=True)
    return adjusted_df

f1_df_adjusted = adjust_features(f1_df, selected_f1, features_f1, features_f2) if not f1_df.empty else pd.DataFrame(columns=features_f1)
f2_df_adjusted = adjust_features(f2_df, selected_f2, features_f1, features_f2) if not f2_df.empty else pd.DataFrame(columns=features_f1)

# ------------------ Model Training ------------------
print("\nTraining Models...")
# Prepare data for training
train_df = df_rounds.dropna(subset=features + target_cols)
X = train_df[features]
y = train_df[target_cols]
rounds = train_df['Round']

# Feature Selection (across all rounds)
initial_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
initial_model.fit(X, y['SigStr1_landed'])
feature_importance = pd.Series(initial_model.feature_importances_, index=features)
top_features = feature_importance.sort_values(ascending=False)[:10].index.tolist()
print(f"Selected Top Features: {top_features}")

# Train models for each round and target
models = {round_num: {} for round_num in range(1, num_rounds + 1)}
predictions = {round_num: {} for round_num in range(1, num_rounds + 1)}
train_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
test_r2 = {round_num: {} for round_num in range(1, num_rounds + 1)}
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [2, 3],
    'subsample': [0.8]
}

# Store test data for total fight plots
test_data = {round_num: {} for round_num in range(1, num_rounds + 1)}
scalers = {round_num: None for round_num in range(1, num_rounds + 1)}

for round_num in range(1, num_rounds + 1):
    print(f"\nTraining models for Round {round_num}...")
    # Filter data for the current round
    round_mask = rounds == round_num
    X_round = X[round_mask][top_features]
    y_round = y[round_mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_round, y_round, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scalers[round_num] = scaler
    test_data[round_num] = {'X_test': X_test, 'y_test': y_test, 'X_test_scaled': X_test_scaled}
    
    # Train models for each target
    for target in tqdm(target_cols, desc=f"Training Round {round_num} models"):
        gb = GradientBoostingRegressor(
            random_state=42,
            n_iter_no_change=10,
            validation_fraction=0.1,
            tol=1e-4
        )
        grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train[target])
        models[round_num][target] = grid_search.best_estimator_
        predictions[round_num][target] = grid_search.best_estimator_.predict(X_test_scaled)
        # Compute training and test R²
        train_r2[round_num][target] = r2_score(y_train[target], grid_search.best_estimator_.predict(X_train_scaled))
        test_r2[round_num][target] = r2_score(y_test[target], predictions[round_num][target])
        print(f"Best Parameters for {target} (Round {round_num}): {grid_search.best_params_}")

    # ------------------ Evaluation Metrics ------------------
    print(f"\nModel Performance for Round {round_num} (Training vs. Test):")
    for target in target_cols:
        train_mae = mean_absolute_error(y_train[target], models[round_num][target].predict(X_train_scaled))
        test_mae = mean_absolute_error(y_test[target], predictions[round_num][target])
        train_mse = mean_squared_error(y_train[target], models[round_num][target].predict(X_train_scaled))
        test_mse = mean_squared_error(y_test[target], predictions[round_num][target])
        print(f"{target} Prediction:")
        print(f"Training MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")
        print(f"Training MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
        print(f"Training Accuracy (R²): {train_r2[round_num][target]:.2f}, Test Accuracy (R²): {test_r2[round_num][target]:.2f}")
        print(f"Overfitting Indicator (Train-Test R² Gap): {(train_r2[round_num][target] - test_r2[round_num][target]):.2f}")

# ------------------ Fight Simulation ------------------
print(f"\nFight Simulation: {selected_f1} vs {selected_f2}")
if not f1_df_adjusted.empty and not f2_df_adjusted.empty:
    f1_input = f1_df_adjusted.mean()
    f2_input = f2_df_adjusted.mean()
    input_diff = pd.Series(index=top_features)
    for f1, diff in zip(features_f1[:6], ['SVR_Diff', 'DefEff_Diff', 'OAT_Diff', 'KDR_Diff', 'SubAgg_Diff', 'FCI_Diff']):
        if diff in top_features:
            input_diff[diff] = f1_input[f1] - f2_input[f1]
    for feature in ['TD_Diff', 'Exp_Diff', 'Age_Diff', 'ReachAdv', 'HeightAdv']:
        if feature in top_features:
            input_diff[feature] = f1_df[feature].mean() if feature in f1_df.columns else 0
    for feature in ['SigStr1_efficiency', 'Head1_efficiency', 'Body1_efficiency', 'Leg1_efficiency']:
        if feature in top_features:
            input_diff[feature] = f1_input[feature] if feature in f1_input.index else 0
    for feature in ['SigStr2_efficiency', 'Head2_efficiency', 'Body2_efficiency', 'Leg2_efficiency']:
        if feature in top_features:
            input_diff[feature] = f2_input[feature.replace('2', '1')] if feature.replace('2', '1') in f2_input.index else 0

    input_row = input_diff.values.reshape(1, -1)
    total_predictions = {target: 0 for target in target_cols}
    for round_num in range(1, num_rounds + 1):
        input_scaled = scalers[round_num].transform(input_row)
        print(f"\nRound {round_num} Predictions:")
        for target in target_cols:
            pred = models[round_num][target].predict(input_scaled)[0]
            total_predictions[target] += pred
            print(f"{target}: {pred:.2f}")

    print(f"\nTotal Fight Predictions:")
    for target in target_cols:
        print(f"{target}: {total_predictions[target]:.2f}")
else:
    print("Insufficient data for simulation.")

# ------------------ Plots ------------------
# Learning Curve Plotting Function
def plot_learning_curve(estimator, X, y, title, filename):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error')
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training MSE')
    plt.plot(train_sizes, test_scores_mean, label='Validation MSE')
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('Mean Squared Error')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

# Plot for each round and total fight
for round_num in range(1, num_rounds + 1):
    print(f"\nGenerating plots for Round {round_num}...")
    X_round = X[rounds == round_num][top_features]
    y_round = y[rounds == round_num]
    X_train = test_data[round_num]['X_test']
    y_train = test_data[round_num]['y_test']
    X_train_scaled = test_data[round_num]['X_test_scaled']
    
    # Learning Curves
    for target in target_cols:
        plot_learning_curve(
            models[round_num][target], 
            X_train_scaled, 
            y_train[target], 
            f'Learning Curve {target} (Round {round_num})', 
            f'learning_curve_round{round_num}_{target}.png'
        )
    
    # Predicted vs Actual
    plt.figure(figsize=(10, 6))
    for target in target_cols:
        plt.scatter(y_train[target], predictions[round_num][target], alpha=0.5, label=target)
    plt.plot([y_train.min().min(), y_train.max().max()], [y_train.min().min(), y_train.max().max()], 'r--')
    plt.xlabel('Actual Strikes')
    plt.ylabel('Predicted Strikes')
    plt.title(f'Predicted vs Actual Strikes (Round {round_num})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'predicted_vs_actual_round{round_num}.png')
    plt.show()
    
    # Residual Plot
    plt.figure(figsize=(10, 6))
    for target in target_cols:
        residuals = y_train[target] - predictions[round_num][target]
        sns.histplot(residuals, kde=True, label=target, alpha=0.5)
    plt.title(f'Residual Distribution (Round {round_num})')
    plt.xlabel('Residuals')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'residual_distribution_round{round_num}.png')
    plt.show()
    
    # Feature Importance
    plt.figure(figsize=(12, 8))
    for target in target_cols:
        feature_importance = pd.Series(models[round_num][target].feature_importances_, index=top_features)
        feature_importance.sort_values(ascending=False)[:10].plot(kind='bar', alpha=0.5, label=target)
    plt.title(f'Top 10 Feature Importances (Round {round_num})')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'feature_importance_round{round_num}.png')
    plt.show()

# Total Fight Plots
print("\nGenerating plots for Total Fight...")
# Aggregate predictions and actuals across rounds
total_actual = pd.DataFrame({target: 0 for target in target_cols}, index=test_data[1]['y_test'].index)
total_pred = pd.DataFrame({target: 0 for target in target_cols}, index=test_data[1]['y_test'].index)
for round_num in range(1, num_rounds + 1):
    for target in target_cols:
        total_actual[target] += test_data[round_num]['y_test'][target]
        total_pred[target] += predictions[round_num][target]

# Use Round 1 data and model for learning curves (as a representative)
X_round1 = X[rounds == 1][top_features]
y_round1 = y[rounds == 1]
X_train, X_test, y_train, y_test = train_test_split(X_round1, y_round1, test_size=0.2, random_state=42)
X_train_scaled = scalers[1].fit_transform(X_train)
X_test_scaled = scalers[1].transform(X_test)

# Learning Curves for Total
for target in target_cols:
    plot_learning_curve(
        models[1][target], 
        X_train_scaled, 
        y_train[target], 
        f'Learning Curve {target} (Total Fight)', 
        f'learning_curve_total_{target}.png'
    )

# Predicted vs Actual (Total)
plt.figure(figsize=(10, 6))
for target in target_cols:
    plt.scatter(total_actual[target], total_pred[target], alpha=0.5, label=target)
plt.plot([total_actual.min().min(), total_actual.max().max()], [total_actual.min().min(), total_actual.max().max()], 'r--')
plt.xlabel('Actual Strikes')
plt.ylabel('Predicted Strikes')
plt.title('Predicted vs Actual Strikes (Total Fight)')
plt.legend()
plt.grid(True)
plt.savefig('predicted_vs_actual_total.png')
plt.show()

# Residual Plot (Total)
plt.figure(figsize=(10, 6))
for target in target_cols:
    residuals = total_actual[target] - total_pred[target]
    sns.histplot(residuals, kde=True, label=target, alpha=0.5)
plt.title('Residual Distribution (Total Fight)')
plt.xlabel('Residuals')
plt.legend()
plt.grid(True)
plt.savefig('residual_distribution_total.png')
plt.show()

# Feature Importance (Total, using Round 1 models)
plt.figure(figsize=(12, 8))
for target in target_cols:
    feature_importance = pd.Series(models[1][target].feature_importances_, index=top_features)
    feature_importance.sort_values(ascending=False)[:10].plot(kind='bar', alpha=0.5, label=target)
plt.title('Top 10 Feature Importances (Total Fight)')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('feature_importance_total.png')
plt.show()