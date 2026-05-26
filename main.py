import pandas as pd
import os

def preparedata():
    data_dir = 'data'
    
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

def matrix_analysis(df):
    print("\n--- Starting Data Analysis ---")
    
    # 1. Reorder columns to put open-ended questions at the end
    col1 = 'Is there anything else you would like to share about your experience with MASTER XR?  '
    col2 = 'What did you like most about the MASTER XR Educational Scene?  '
    
    cols_to_move = [col for col in [col1, col2] if col in df.columns]
    other_cols = [col for col in df.columns if col not in cols_to_move]
    df = df[other_cols + cols_to_move]
    
    print("Columns reordered: open-ended questions moved to the end.")
    
    # 2. Analyze demographics: Female percentage
    if 'Gender:' in df.columns:
        genders = df['Gender:'].dropna().astype(str).str.strip().str.lower()
        total_valid = len(genders)
        female_count = (genders == 'female').sum()
        
        if total_valid > 0:
            female_percentage = (female_count / total_valid) * 100
            print("\n[Demographics]")
            print(f"- Total participants (valid gender): {total_valid}")
            print(f"- Female participants: {female_count}")
            print(f"- Female percentage: {female_percentage:.2f}%")
        else:
            print("No valid gender data found.")
    else:
        print("Column 'Gender:' not found.")
        
    return df

if __name__ == "__main__":
    # Prepare the data
    matrix = preparedata()
    
    # Perform analysis and reorder columns
    matrix = matrix_analysis(matrix)
    
    # Save the final reordered matrix back without the .0
    matrix_path = os.path.join('data', 'merged_matrix.csv')
    
    # Convert float columns that are whole numbers to 'Int64' to avoid '.0' in output
    for col in matrix.columns:
        if pd.api.types.is_float_dtype(matrix[col]):
            # Check if all non-NaN values are integers
            if matrix[col].dropna().apply(lambda x: x.is_integer()).all():
                matrix[col] = matrix[col].astype('Int64')
                
    matrix.to_csv(matrix_path, index=False)
    print(f"\nFinal matrix successfully saved to {matrix_path}")
