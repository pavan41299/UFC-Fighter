import os
import glob
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# Helper function to expand list columns into multiple columns with given column names.
def expand_list_column(df, column_name, new_columns):
    if column_name in df.columns and not df[column_name].isnull().all():
        expanded = pd.DataFrame(df[column_name].tolist(), columns=new_columns)
        df = df.drop(columns=[column_name]).reset_index(drop=True)
        df = pd.concat([df, expanded], axis=1)
    return df

# Directory containing your HTML files.
input_directory = r'ufc_pages'

# Prepare lists for aggregated data.
all_events = []
all_fighters = []
all_metadata = []
all_judge_scores = []
all_fight_totals = []           
all_sig_strikes = []            
all_round_fight_totals = []     
all_round_sig_strikes = []      
all_charts_data = []

# Initialize fight counter.
fight_num = 1

# Get list of all HTML files.
file_list = glob.glob(os.path.join(input_directory, '*.html'))

# Loop over each HTML file in the directory with a progress bar.
for file_path in tqdm(file_list, desc="Processing Files", unit="file"):
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    # --- Extract event details ---
    event_tag = soup.select_one('h2.b-content__title a')
    event_name = event_tag.text.strip() if event_tag else "N/A"
    event_link = event_tag['href'] if event_tag and event_tag.has_attr('href') else "N/A"

    # Extract event date to get the year
    event_date = None
    for item in soup.select('.b-fight-details__text-item'):
        label = item.select_one('.b-fight-details__label')
        if label and "Date" in label.text:
            event_date = item.text.replace(label.text, '').strip()
            break  # Stop searching after finding the date

    event_year = None
    if event_date:
        try:
            event_year = pd.to_datetime(event_date, errors='coerce').year
        except:
            event_year = "Unknown"

    all_events.append({
        'fight_num': fight_num,
        'File': os.path.basename(file_path),
        'Event Name': event_name,
        'Event Link': event_link,
        'Event Year': event_year  # Add extracted year
    })
    
    # --- Extract fighter details ---
    for person in soup.select('.b-fight-details__person'):
        name = person.select_one('h3 a').text.strip() if person.select_one('h3 a') else "Unknown"
        link = person.select_one('h3 a')['href'] if person.select_one('h3 a') and person.select_one('h3 a').has_attr('href') else "N/A"
        status = person.select_one('.b-fight-details__person-status').text.strip() if person.select_one('.b-fight-details__person-status') else "Unknown"
        nickname = person.select_one('.b-fight-details__person-title').text.strip() if person.select_one('.b-fight-details__person-title') else ""
        all_fighters.append({
            'fight_num': fight_num,
            'File': os.path.basename(file_path),
            'Event Name': event_name,
            'Fighter Name': name,
            'Link': link,
            'Result': status,
            'Nickname': nickname
        })
    
    # --- Extract fight metadata ---
    metadata = {
        'fight_num': fight_num, 
        'File': os.path.basename(file_path), 
        'Event Name': event_name,
        'Event Year': event_year  # Include year in metadata
    }

    for item in soup.select('.b-fight-details__text-item'):
        label = item.select_one('.b-fight-details__label')
        if label:
            key = label.text.strip().replace(':', '')
            value = item.text.replace(label.text, '').strip()
            metadata[key] = value

    all_metadata.append(metadata)
    
    # --- Extract judges' scores ---
    for i in soup.select('p.b-fight-details__text i'):
        judge_name_tag = i.select_one('span')
        if judge_name_tag:
            score = i.text.replace(judge_name_tag.text, '').strip()
            all_judge_scores.append({
                'fight_num': fight_num,
                'File': os.path.basename(file_path),
                'Event Name': event_name,
                'Judge': judge_name_tag.text.strip(),
                'Score': score
            })
    
    # --- Extract overall fight totals ---
    total_tables = soup.select('section.b-fight-details__section.js-fight-section table')
    if total_tables:
        for row in total_tables[0].select('tr.b-fight-details__table-row'):
            cols = [col.text.strip() for col in row.select('p.b-fight-details__table-text')]
            if cols:
                all_fight_totals.append({
                    'fight_num': fight_num,
                    'File': os.path.basename(file_path),
                    'Event Name': event_name,
                    'Fight Totals': cols
                })
    
    # --- Extract round-wise fight totals ---
    per_round_tables = soup.select('table.b-fight-details__table.js-fight-table')
    if len(per_round_tables) > 0:
        round_totals = [[col.text.strip() for col in row.select('p.b-fight-details__table-text')]
                        for row in per_round_tables[0].select('tr.b-fight-details__table-row') if row.select('p')]
        for rt in round_totals:
            all_round_fight_totals.append({
                'fight_num': fight_num,
                'File': os.path.basename(file_path),
                'Event Name': event_name,
                'Round Fight Totals': rt
            })
    
    # Increment fight number for next file.
    fight_num += 1

# Convert aggregated lists to DataFrames.
df_events = pd.DataFrame(all_events)
df_fighters = pd.DataFrame(all_fighters)
df_metadata = pd.DataFrame(all_metadata)
df_judge_scores = pd.DataFrame(all_judge_scores)
df_fight_totals = pd.DataFrame(all_fight_totals)
df_round_fight_totals = pd.DataFrame(all_round_fight_totals)

# Save DataFrames to CSV files.
df_events.to_csv('all_events.csv', index=False)
df_fighters.to_csv('all_fighters.csv', index=False)
df_metadata.to_csv('all_metadata.csv', index=False)
df_judge_scores.to_csv('all_judge_scores.csv', index=False)
df_fight_totals.to_csv('all_fight_totals.csv', index=False)
df_round_fight_totals.to_csv('all_round_fight_totals.csv', index=False)

# Print a summary to the console.
print("=== Summary of Extracted Data ===")
print("Total Files Processed:", fight_num - 1)
print("Total Events:", len(df_events))
print("Total Fighters:", len(df_fighters))
print("Total Metadata Records:", len(df_metadata))
print("Total Judges' Score Records:", len(df_judge_scores))
print("Total Fight Totals Records:", len(df_fight_totals))
print("Total Round-wise Fight Totals:", len(df_round_fight_totals))
print("✅ All details extracted, flattened into separate columns with proper headers, and saved to CSV files.")
