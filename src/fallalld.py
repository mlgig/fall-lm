import numpy as np
import pandas as pd
from tqdm import tqdm

def magnitude(arr):
    x, y, z = arr.T.astype('float')
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    magnitude -= min(magnitude)
    return magnitude

def load_fallalld(clip=False):
    fallalld = pd.read_pickle("data/FallAllD.pkl")
    fallalld_waist = fallalld[fallalld['Device']=='Waist']
    fallalld_waist = fallalld_waist.reset_index().drop(columns=['index'])
    ADL_ids = [i for i in range(13,43)]
    fall_ids = [i for i in range(101,136)]
    adls_dict = {id: 0 for id in ADL_ids}
    falls_dict = {id: 1 for id in fall_ids}
    activity_dict = {**adls_dict, **falls_dict}
    fallalld_waist['target'] = fallalld_waist['ActivityID'].replace(activity_dict)
    # drop rows with activities outside the chosen ones
    fallalld_waist.drop(fallalld_waist[fallalld_waist['target']>1].index, inplace=True)
    fallalld_waist.drop(columns=['Gyr', 'Mag', 'Bar', 'TrialNo', 'Device'], inplace=True)
    fallalld_waist['accel_g'] = fallalld_waist['Acc'].apply(
         g_from_LSB).apply(magnitude)
    if clip:
         fallalld_waist['accel_g'] = fallalld_waist['accel_g'].apply(clip_arr)
    return fallalld_waist

def clip_arr(arr):
    return np.clip(arr, -2,2)

def g_from_LSB(arr, sensitivity=0.244):
    # Acceleration (g) = Raw data (LSB) * Sensitivity (mg/LSB) / 1000 (mg/g)
	return (arr * sensitivity)/1000

def reshape_arr(arr):
	return np.reshape(arr, (1,-1))

def get_X_y(df, winsize=12, clip=False):
    freq = 238
    X = np.zeros([df.shape[0], winsize*freq])
    # start 1 sec before the fall
    start = int(df.reset_index()['accel_g'][0].size/2) - int(winsize*freq/2)
    end = start + (freq * winsize)
    for i, row in enumerate(df['accel_g']):
        this_row = row[:, start:end]
        if this_row.shape[1] != winsize*freq:
             continue
        if clip:
             this_row = np.clip(this_row, 0, 8)
        X[i] = this_row
    y = np.array(df['target'], dtype='uint8')
    return X, y

def adapter_fallalld_to_costream(df):
    """
    Converts the FallAllD summary DataFrame into a Costream Subject Map.
    Args:
        df: The DataFrame
    Returns:
        subject_map: Dict {subject_id: [df_recording_1, df_recording_2, ...]}
    """
    
    subject_map = {}
    print("Adapting DataFrame to Costream format...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        subj_id = row['SubjectID']
        
        signal = np.array(row['accel_g'], dtype=np.float32)
        rec_df = pd.DataFrame({'mag': signal})
        rec_df['label'] = 0
        if row['target'] == 1: # It's a Fall
            # Heuristic: Mark the Peak Magnitude as the Fall Point
            peak_idx = np.argmax(signal)
            rec_df.loc[peak_idx, 'label'] = 1
        if subj_id not in subject_map:
            subject_map[subj_id] = []
        subject_map[subj_id].append(rec_df)
        
    return subject_map