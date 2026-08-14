"""Train final production models on the full dataset and persist artifacts."""
import pickle
import sys
sys.path.insert(0, 'src')
from data_pipeline import build_dataset
from models import train_ml_model

df = build_dataset('/mnt/user-data/uploads/HHS_Unaccompanied_Alien_Children_Program.csv')
df.to_csv('data/featured_dataset.csv', index=False)

for target in ['hhs_care', 'discharged']:
    for mtype, name in [('rf', 'rf'), ('gb', 'gb')]:
        model, feat_cols, resid_std = train_ml_model(df, target, mtype)
        with open(f'models/{name}_{target}.pkl', 'wb') as f:
            pickle.dump({'model': model, 'feat_cols': feat_cols, 'resid_std': resid_std}, f)
        print(f"Saved models/{name}_{target}.pkl  (resid_std={resid_std:.2f})")

print("Done.")
