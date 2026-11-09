import pandas as pd, shutil

path = "/mnt/ceph/mjain/software/envs/bio/stems/human_rRNA/stems2_human_rRNA.csv"
df = pd.read_csv(path)

mask = df["name"].str.contains("16S|23S", case=False, regex=True, na=False)
print("Removing:", df[mask]["name"].unique())

shutil.copy(path, path + ".bak")
df[~mask].to_csv(path, index=False)
print(f"Kept {(~mask).sum()} rows, removed {mask.sum()}")