import pandas as pd
import os
import math

def preparedata():
    data_dir = 'data/original data/question'
    
    # 1. Load ICETeachStudySubjects
    subjects_path = os.path.join(data_dir, 'ICETeachStudySubjects.csv')
    try:
        # Try loading with semicolon first (standard for European CSVs)
        df_subjects = pd.read_csv(subjects_path, sep=';', encoding='latin-1')
        if len(df_subjects.columns) < 3:
            df_subjects = pd.read_csv(subjects_path, sep=',', encoding='latin-1')
    except Exception:
        df_subjects = pd.read_csv(subjects_path, sep=',', encoding='utf-8')
        
    # Remove empty columns (Unnamed and all NaN)
    df_subjects.dropna(how='all', axis=1, inplace=True)
    unnamed_cols = [c for c in df_subjects.columns if 'unnamed' in str(c).lower()]
    df_subjects.drop(columns=unnamed_cols, inplace=True, errors='ignore')
    
    # Remove name, email, gender, date, ora (case-insensitive)
    cols_to_drop_sub = [c for c in df_subjects.columns if str(c).strip().lower() in ['name', 'email', 'gender', 'date', 'ora']]
    df_subjects.drop(columns=cols_to_drop_sub, inplace=True, errors='ignore')
    
    # Find the ID column (Subject) and rename it to 'Participant ID'
    if 'Subject' in df_subjects.columns:
        df_subjects.rename(columns={'Subject': 'Participant ID'}, inplace=True)
    elif 'Participant ID:' in df_subjects.columns:
        df_subjects.rename(columns={'Participant ID:': 'Participant ID'}, inplace=True)
        
    df_subjects['Participant ID'] = pd.to_numeric(df_subjects['Participant ID'], errors='coerce')
    df_subjects.dropna(subset=['Participant ID'], inplace=True)
    
    # Remove rows without information (e.g. rows with only the ID or few data points)
    df_subjects.dropna(thresh=3, inplace=True)
    
    # Remove invalid IDs caused by malformed rows in the original CSV
    df_subjects = df_subjects[df_subjects['Participant ID'] <= 30]
    
    # Remove Participant ID 2 as requested
    df_subjects = df_subjects[df_subjects['Participant ID'] != 2]
    
    # 2. Load ICETeach_Teach
    teach_path = os.path.join(data_dir, 'ICETeach_Teach.csv')
    df_teach = pd.read_csv(teach_path, sep=',', encoding='utf-8')
    df_teach.dropna(how='all', axis=1, inplace=True)
    
    # Remove Informazioni cronologiche and Consent
    cols_to_drop_teach = [c for c in df_teach.columns if 'informazioni cronologiche' in str(c).lower() or 'consent' in str(c).lower()]
    df_teach.drop(columns=cols_to_drop_teach, inplace=True, errors='ignore')
    
    if 'Participant ID:' in df_teach.columns:
        df_teach.rename(columns={'Participant ID:': 'Participant ID'}, inplace=True)
    
    df_teach['Participant ID'] = pd.to_numeric(df_teach['Participant ID'], errors='coerce')
    df_teach.dropna(subset=['Participant ID'], inplace=True)

    # 3. Load ICETeach_Viroo
    viroo_path = os.path.join(data_dir, 'ICETeach_Viroo.csv')
    df_viroo = pd.read_csv(viroo_path, sep=',', encoding='utf-8')
    df_viroo.dropna(how='all', axis=1, inplace=True)
    
    # Remove Informazioni cronologiche and system specifications
    cols_to_drop_viroo = [c for c in df_viroo.columns if 'informazioni cronologiche' in str(c).lower() or 'system specifications' in str(c).lower()]
    df_viroo.drop(columns=cols_to_drop_viroo, inplace=True, errors='ignore')
    
    if 'Participant ID:' in df_viroo.columns:
        df_viroo.rename(columns={'Participant ID:': 'Participant ID'}, inplace=True)
        
    df_viroo['Participant ID'] = pd.to_numeric(df_viroo['Participant ID'], errors='coerce')
    df_viroo.dropna(subset=['Participant ID'], inplace=True)

    # 4. Merge matrices (using ID as the main key)
    # Left join starting from subjects
    merged_matrix = df_subjects.merge(df_teach, on='Participant ID', how='left', suffixes=('', '_Teach'))
    merged_matrix = merged_matrix.merge(df_viroo, on='Participant ID', how='left', suffixes=('', '_Viroo'))
    
    # Remove any duplicate columns that may have been generated or completely empty columns
    merged_matrix.dropna(how='all', axis=1, inplace=True)
    
    # Ensure ID is an integer and sort
    merged_matrix['Participant ID'] = merged_matrix['Participant ID'].astype(int)
    merged_matrix.sort_values(by='Participant ID', inplace=True)
    
    # 5. Export
    return merged_matrix


def parse_position(pos_str):
    """Helper function to parse the position tuple string into a float tuple."""
    try:
        clean_str = pos_str.replace('(', '').replace(')', '')
        parts = clean_str.split(',')
        if len(parts) == 3:
            return (float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip()))
    except Exception:
        pass
    return None


def log_analysis(df, start_time_sec=120.0):
    print(f"\n--- Starting Log Analysis (skipping first {start_time_sec}s) ---")
    
    log_dir = os.path.join('data', 'original data', 'log')
    if not os.path.exists(log_dir):
        print(f"Log directory not found at {log_dir}")
        return df
        
    # Initialize new columns with NA
    df['Total_Distance_m'] = pd.NA
    df['Interactions_Count'] = pd.NA
    df['Most_Looked_Machine'] = pd.NA
    df['Most_Interacted_Machine'] = pd.NA
    df['Total_Time_s'] = pd.NA
    
    global_interacted_objects = {}
    global_looked_objects = {}
    
    for index, row in df.iterrows():
        participant_id = row.get('Participant ID')
        if pd.isna(participant_id):
            continue
            
        try:
            p_id = int(participant_id)
        except ValueError:
            continue
        
        # 1. Parse Interactions log
        interactions_file = os.path.join(log_dir, f'Interactions_{p_id}.txt')
        if os.path.exists(interactions_file):
            interactions_count = 0
            interacted_objects = {}
            try:
                with open(interactions_file, 'r', encoding='utf-8') as f:
                    current_time = 0.0
                    for line in f:
                        if line.startswith('Time:'):
                            parts = line.split('|')
                            time_str = parts[0].replace('Time:', '').strip().replace(',', '.')
                            try:
                                current_time = float(time_str)
                            except ValueError:
                                pass
                                
                            if current_time >= start_time_sec:
                                for part in parts:
                                    part = part.strip()
                                    if part.startswith('Interacted Panel:'):
                                        panel = part.replace('Interacted Panel:', '').strip()
                                        if panel:
                                            interacted_objects[panel] = interacted_objects.get(panel, 0) + 1
                                            global_interacted_objects[panel] = global_interacted_objects.get(panel, 0) + 1
                                            
                        elif 'User said:' in line:
                            if current_time >= start_time_sec:
                                interactions_count += 1
                
                df.at[index, 'Interactions_Count'] = interactions_count
                
                most_interacted = 'None'
                if interacted_objects:
                    most_interacted = max(interacted_objects, key=interacted_objects.get)
                df.at[index, 'Most_Interacted_Machine'] = most_interacted
                
            except Exception as e:
                print(f"Error reading {interactions_file}: {e}")
        
        # 2. Parse Movement and Gaze log
        movement_file = os.path.join(log_dir, f'MovementGaze_{p_id}.txt')
        if os.path.exists(movement_file):
            total_distance = 0.0
            total_time = 0.0
            looked_objects = {}
            try:
                with open(movement_file, 'r', encoding='utf-8') as f:
                    prev_pos = None
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                            
                        parts = line.split('|')
                        if len(parts) == 0:
                            continue
                            
                        # Extract and check time
                        time_str = parts[0].replace('Time:', '').strip().replace(',', '.')
                        try:
                            current_time = float(time_str)
                            total_time = max(total_time, current_time)
                        except ValueError:
                            continue
                            
                        if current_time < start_time_sec:
                            continue
                            
                        pos = None
                        looking_to = None
                        
                        for part in parts:
                            part = part.strip()
                            if part.startswith('Pos:'):
                                pos_str = part.replace('Pos:', '').strip()
                                pos = parse_position(pos_str)
                            elif part.startswith('Looking To:'):
                                looking_to = part.replace('Looking To:', '').strip()
                                
                        if pos is not None:
                            if prev_pos is not None:
                                dist = math.sqrt(sum((a - b)**2 for a, b in zip(pos, prev_pos)))
                                total_distance += dist
                            prev_pos = pos
                            
                        if looking_to is not None and looking_to.lower() != 'no object' and looking_to != '':
                            looked_objects[looking_to] = looked_objects.get(looking_to, 0) + 1
                            global_looked_objects[looking_to] = global_looked_objects.get(looking_to, 0) + 1
                            
                most_looked = 'None'
                if looked_objects:
                    most_looked = max(looked_objects, key=looked_objects.get)
                    
                df.at[index, 'Total_Distance_m'] = round(total_distance, 2)
                df.at[index, 'Total_Time_s'] = round(total_time, 2)
                df.at[index, 'Most_Looked_Machine'] = most_looked
                
                print(f"Participant {p_id}: Time={total_time:.2f}s, Dist={total_distance:.2f}m, Interactions={df.at[index, 'Interactions_Count']}, Most Looked={most_looked}")
                
            except Exception as e:
                print(f"Error reading {movement_file}: {e}")
        
    # After processing all logs, calculate statistics by modality
    print("\n--- Log Analysis Statistics by Modality ---")
    if 'modality' in df.columns:
        temp_df = df.copy()
        temp_df['Total_Distance_m'] = pd.to_numeric(temp_df['Total_Distance_m'], errors='coerce')
        temp_df['Interactions_Count'] = pd.to_numeric(temp_df['Interactions_Count'], errors='coerce')
        temp_df['Total_Time_s'] = pd.to_numeric(temp_df['Total_Time_s'], errors='coerce')
        
        metrics = ['Total_Distance_m', 'Interactions_Count', 'Total_Time_s']
        grouped = temp_df.groupby('modality')[metrics].mean(numeric_only=True).round(2)
        grouped.reset_index(inplace=True)
        print(grouped.to_string(index=False))
        
        out_path = os.path.join('data', 'log_analysis_by_modality.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Log statistics saved to: {out_path}")
        
    print("\n--- Global Most Interacted Machines ---")
    if global_interacted_objects:
        interacted_df = pd.DataFrame(list(global_interacted_objects.items()), columns=['Machine', 'Total_Interactions'])
        interacted_df.sort_values(by='Total_Interactions', ascending=False, inplace=True)
        print(interacted_df.to_string(index=False))
        interacted_out = os.path.join('data', 'global_interacted_machines.csv')
        interacted_df.to_csv(interacted_out, index=False)
        print(f"Global interacted machines saved to: {interacted_out}")

    print("\n--- Global Most Looked Machines ---")
    if global_looked_objects:
        looked_df = pd.DataFrame(list(global_looked_objects.items()), columns=['Machine', 'Total_Looks_Frames'])
        looked_df.sort_values(by='Total_Looks_Frames', ascending=False, inplace=True)
        # Convert frames to approximate seconds (assuming 1 frame = 0.01s as seen in logs)
        looked_df['Approx_Total_Time_s'] = looked_df['Total_Looks_Frames'] * 0.01
        print(looked_df.head(10).to_string(index=False)) # print top 10
        looked_out = os.path.join('data', 'global_looked_machines.csv')
        looked_df.to_csv(looked_out, index=False)
        print(f"Global looked machines saved to: {looked_out}")
        
    return df


def create_evaluation_table(df):
    print("\n--- Creating Simplified Evaluation Table ---")
    
    # 1. Select basic demographic and modality columns
    base_cols = [
        'Participant ID', 'nation', 'modality', 'Test modality', 
        'Gender:', 'Age:', 'Highest level of education:', 'Current occupation:  ',
        'Experience Gaming',
        'How familiar you are with Industrial environment and processes?',
        'How familiar you are with Industrial robotics handling or programming?'
    ]
    
    selected_cols = [c for c in base_cols if c in df.columns]
    eval_df = df[selected_cols].copy()
    
    # 2. Define the correct answers mapping
    correct_answers = {
        "How can the Meta-MES handle the detection of a defective piece made by the quality control cell? ": "can perform an automatic replanning of the procedure replacing the piece",
        "In the Subtractive Manufacturing cell, what is the role of the UR5 robot? ": "it loads and unloads parts into the emco milling machine",
        "The Assembly cell (Cell 4) uses which two types of robotic arms? ": "abb and kuka",
        "The Quality Control cell detects defects based on which criteria?": "color, shape, and geometry",
        "In the context of the Assembly Cell, what is the role of the OPC-UA server? ": "it encapsulates internal complexity and exposes services to the meta-mes",
        "The VERTIMAG EF system is associated with which facility feature? ": "the automated warehouse",
        "Which is the tallest machine in the lab?": "automatic warehouse",
        " Which is the dual arm robot?": "abb",
        "How many bays are there in the transport line": "3",
        "Which machine is closer to the SPEA testing machine?": "milling machine",
        "How many pallets are on the conveyor?": "10"
    }
    
    # 3. Evaluate each question
    for col, correct_ans in correct_answers.items():
        if col in df.columns:
            clean_correct = correct_ans.lower().replace('.', '').strip()
            eval_df[col] = df[col].astype(str).str.lower().str.replace('.', '', regex=False).str.strip() == clean_correct
        else:
            print(f"Warning: Question column not found: '{col}'")
            
    # 4. Compute scores
    audioguide_cols = [
        "How can the Meta-MES handle the detection of a defective piece made by the quality control cell? ",
        "In the Subtractive Manufacturing cell, what is the role of the UR5 robot? ",
        "The Assembly cell (Cell 4) uses which two types of robotic arms? ",
        "The Quality Control cell detects defects based on which criteria?",
        "In the context of the Assembly Cell, what is the role of the OPC-UA server? ",
        "The VERTIMAG EF system is associated with which facility feature? "
    ]
    
    presence_cols = [
        "Which is the tallest machine in the lab?",
        " Which is the dual arm robot?",
        "How many bays are there in the transport line",
        "Which machine is closer to the SPEA testing machine?",
        "How many pallets are on the conveyor?"
    ]
    
    valid_audioguide = [c for c in audioguide_cols if c in eval_df.columns]
    valid_presence = [c for c in presence_cols if c in eval_df.columns]
    
    if valid_audioguide:
        eval_df['Audioguide_score'] = eval_df[valid_audioguide].sum(axis=1) / len(valid_audioguide)
    else:
        print("Warning: Could not compute Audioguide_score (columns missing).")
        
    if valid_presence:
        eval_df['spatial_awareness_score'] = eval_df[valid_presence].sum(axis=1) / len(valid_presence)
    else:
        print("Warning: Could not compute spatial_awareness_score (columns missing).")
            
    # 5. Save the new evaluation matrix
    out_path = os.path.join('data', 'simplified_evaluation.csv')
    eval_df.to_csv(out_path, index=False)
    print(f"Simplified evaluation table successfully saved to: {out_path}")
    
    return eval_df

def analyze_all_categories(eval_df):
    categories = [
        'modality',
        'Experience Gaming',
        'Highest level of education:',
        'How familiar you are with Industrial environment and processes?',
        'How familiar you are with Industrial robotics handling or programming?'
    ]
    
    scores_to_analyze = ['Audioguide_score', 'spatial_awareness_score']
    available_scores = [s for s in scores_to_analyze if s in eval_df.columns]
    
    if not available_scores:
        print("Error: Score columns not found for analysis.")
        return None
        
    for category in categories:
        if category not in eval_df.columns:
            print(f"Warning: '{category}' column not found. Skipping analysis.")
            continue
            
        print(f"\n--- Score Analysis by {category} ---")
        
        # Clean string categories for consistent grouping
        temp_df = eval_df.copy()
        if temp_df[category].dtype == object:
            temp_df[category] = temp_df[category].astype(str).str.strip().str.capitalize()
        
        # Group by category and calculate mean and std
        grouped = temp_df.groupby(category)[available_scores].agg(['mean'])#,std
        
        # Flatten the hierarchical column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped.reset_index(inplace=True)
        
        # Display the results
        print(grouped.to_string(index=False))
        
        # Save to a new CSV file
        safe_name = category.replace(' ', '_').replace('?', '').replace(':', '')
        out_path = os.path.join('data', f'analysis_by_{safe_name}.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Analysis saved to: {out_path}")

def analyze_all_categories_modality(eval_df):
    categories = [
        'modality',
        'Experience Gaming',
        'Highest level of education:',
        'How familiar you are with Industrial environment and processes?',
        'How familiar you are with Industrial robotics handling or programming?'
    ]
    
    scores_to_analyze = ['Audioguide_score', 'spatial_awareness_score']
    available_scores = [s for s in scores_to_analyze if s in eval_df.columns]
    
    if not available_scores:
        print("Error: Score columns not found for analysis.")
        return None
        
    for category in categories:
        if category not in eval_df.columns:
            print(f"Warning: '{category}' column not found. Skipping analysis.")
            continue
            
        print(f"\n--- Score Analysis by {category} ---")
        
        # Clean string categories for consistent grouping
        temp_df = eval_df.copy()
        if temp_df[category].dtype == object:
            temp_df[category] = temp_df[category].astype(str).str.strip().str.capitalize()
        
        # Determine grouping keys to subdivide by modality
        if category == 'modality':
            group_keys = ['modality']
        else:
            group_keys = ['modality', category]
            
        # Group by the keys and calculate mean and std
        grouped = temp_df.groupby(group_keys)[available_scores].agg(['mean'])#,std
        
        # Flatten the hierarchical column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped.reset_index(inplace=True)
        
        # Display the results
        print(grouped.to_string(index=False))
        
        # Save to a new CSV file
        safe_name = category.replace(' ', '_').replace('?', '').replace(':', '')
        out_path = os.path.join('data', f'analysis_by_{safe_name}.csv')
        grouped.to_csv(out_path, index=False)
        print(f"Analysis saved to: {out_path}")


if __name__ == "__main__":
    # Prepare the data
    matrix = preparedata()
    
    # Perform analysis and reorder columns
    #matrix = matrix_analysis(matrix)
    
    # Create the simplified evaluation table for correct answers
    eval_matrix = create_evaluation_table(matrix)

    # Perform log analysis to append distance, gaze, and interactions
    # Skipping the first 120 seconds (2 minutes) where the user is stationary
    matrix = log_analysis(matrix, start_time_sec=180.0)

    # Save the final reordered matrix back without the .0
    matrix_path = os.path.join('data', 'final_matrix.csv')
    
    # Convert float columns that are whole numbers to 'Int64' to avoid '.0' in output
    for col in matrix.columns:
        if pd.api.types.is_float_dtype(matrix[col]):
            # Check if all non-NaN values are integers
            if matrix[col].dropna().apply(lambda x: x.is_integer()).all():
                matrix[col] = matrix[col].astype('Int64')
                
    matrix.to_csv(matrix_path, index=False)
    print(f"\nFinal matrix successfully saved to {matrix_path}")
    
    # Analyze the scores grouped by all specified categories
    #analyze_all_categories(eval_matrix)
    
    # Analyze the scores grouped by all specified categories including modality
    #analyze_all_categories_modality(eval_matrix)
    
    