import pandas as pd
import numpy as np
import pyreadstat
import os
from datetime import datetime, timedelta

# --- Configuration ---
FOLDER_NAME = 'cdrsce_data'
if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)
    print(f"Created folder: {FOLDER_NAME}")

N_SUBJECTS = 20
subjects = [f"001-{str(i).zfill(3)}" for i in range(1, N_SUBJECTS + 1)]
arms = ['Placebo', 'Drug High Dose', 'Drug Low Dose']

print("Generating rich clinical data...")

# --- 1. DM (Demographics) ---
dm = pd.DataFrame({
    'STUDYID': 'XYZ-999', 'DOMAIN': 'DM',
    'USUBJID': subjects,
    'AGE': np.random.randint(25, 75, size=N_SUBJECTS),
    'SEX': np.random.choice(['M', 'F'], size=N_SUBJECTS, p=[0.6, 0.4]),
    'RACE': np.random.choice(['WHITE', 'BLACK', 'ASIAN', 'OTHER'], size=N_SUBJECTS),
    'ARM': np.random.choice(arms, size=N_SUBJECTS),
    'RFSTDTC': '2025-01-01'
})

# --- 2. LB (Labs) & VS (Vitals) - Longitudinal ---
# We need Baseline vs Post-Baseline for shift/waterfall plots
visits = ['Screening', 'Baseline', 'Week 4', 'Week 8', 'End of Study']
lb_data = []
vs_data = []

for sub in subjects:
    # Assign an inherent "health factor" to make data realistic per patient
    subject_factor = np.random.uniform(0.8, 1.2)
    
    baseline_alt = 0
    baseline_sysbp = 0

    for i, vis in enumerate(visits):
        # LB: ALT (Liver enzyme)
        # Simulate slight increase over time for drug arms over placebo
        arm = dm.loc[dm['USUBJID'] == sub, 'ARM'].values[0]
        drug_effect = i * 5 if 'Drug' in arm and i > 1 else 0
        noise = np.random.randint(-5, 10)
        alt_val = int((30 * subject_factor) + drug_effect + noise)
        alt_val = max(10, alt_val) # Ensure no negative values
        
        lb_data.append([sub, vis, i*2, 'ALT', alt_val, 'U/L'])
        if vis == 'Baseline': baseline_alt = alt_val

        # VS: SYSBP (Systolic Blood Pressure)
        sysbp_val = int((120 * subject_factor) + np.random.randint(-10, 15))
        vs_data.append([sub, vis, i*2, 'SYSBP', sysbp_val, 'mmHg'])
        if vis == 'Baseline': baseline_sysbp = sysbp_val

lb = pd.DataFrame(lb_data, columns=['USUBJID', 'VISIT', 'VISITNUM', 'LBTESTCD', 'AVAL', 'LBORRESU'])
vs = pd.DataFrame(vs_data, columns=['USUBJID', 'VISIT', 'VISITNUM', 'VSTESTCD', 'AVAL', 'VSORRESU'])

# --- 3. EX (Exposure) ---
ex_data = []
start_date = datetime.strptime('2025-01-01', '%Y-%m-%d')
for sub in subjects:
    arm = dm.loc[dm['USUBJID'] == sub, 'ARM'].values[0]
    dose = 100 if 'High' in arm else (50 if 'Low' in arm else 0)
    duration = np.random.randint(50, 60)
    end_date = start_date + timedelta(days=duration)
    ex_data.append([sub, arm, dose, 'mg', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), duration])

ex = pd.DataFrame(ex_data, columns=['USUBJID', 'EXTRT', 'EXDOSE', 'EXDOSU', 'EXSTDTC', 'EXENDTC', 'EXDUR'])

# --- 4. DS (Disposition) ---
ds_data = []
for sub in subjects:
    # Introduce some dropouts
    status = np.random.choice(['COMPLETED', 'DISCONTINUED DUE TO AE', 'WITHDRAWAL BY SUBJECT'], p=[0.7, 0.2, 0.1])
    ds_data.append([sub, status, '2025-03-01'])
ds = pd.DataFrame(ds_data, columns=['USUBJID', 'DSDECOD', 'DSSTDTC'])

# --- 5. MH (Medical History) ---
mh_terms = ['Hypertension', 'Type 2 Diabetes', 'Hyperlipidemia', 'GERD', 'Osteoarthritis']
mh_data = []
for sub in subjects:
    # Give each subject 0 to 3 medical history items
    n_mh = np.random.randint(0, 4)
    terms = np.random.choice(mh_terms, size=n_mh, replace=False)
    for term in terms:
        mh_data.append([sub, term, 'Y'])
mh = pd.DataFrame(mh_data, columns=['USUBJID', 'MHTERM', 'MHOCCUR'])

# --- 6. QS (Questionnaires - e.g., Pain Score) ---
qs_data = []
for sub in subjects:
    arm = dm.loc[dm['USUBJID'] == sub, 'ARM'].values[0]
    for i, vis in enumerate(['Baseline', 'Week 8']):
        # Simulate pain improving (score going down) in drug arms
        base_score = np.random.randint(6, 9)
        improvement = np.random.randint(2, 5) if 'Drug' in arm and vis == 'Week 8' else 0
        score = base_score if vis == 'Baseline' else max(0, base_score - improvement)
        qs_data.append([sub, vis, 'PAIN_VAS', score])
qs = pd.DataFrame(qs_data, columns=['USUBJID', 'VISIT', 'QSTESTCD', 'AVAL'])


# --- Export to XPT ---
datasets = {'DM': dm, 'LB': lb, 'VS': vs, 'EX': ex, 'DS': ds, 'MH': mh, 'QS': qs}
for name, df in datasets.items():
    file_path = os.path.join(FOLDER_NAME, f"{name.lower()}.xpt")
    pyreadstat.write_xport(df, file_path, table_name=name)
    print(f"  -> Generated {name}: {len(df)} records")

print("\nData generation complete. XPT files are in the 'cdrsce_data' folder.")