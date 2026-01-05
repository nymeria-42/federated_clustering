import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
DATASET_PATH = "/home/nymeria/repos/federated_clustering/fed-clustering/des_dr2_sample_dec_-58_to_-56.csv"  # Set your dataset path
N_CLIENTS = 10
N_CLUSTERS = 3  # Set your number of clusters
FEATURE_COLUMNS = [
    "coadd_object_id",
    # "mag_auto_g_dered",
    "mag_auto_r_dered",
    "mag_auto_i_dered",
    "mag_auto_z_dered",
    "mag_auto_y_dered",
]

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
# here calculate colors as differences of magnitudes
df["gmr"] = df["mag_auto_g_dered"] - df["mag_auto_r_dered"]
df["rmi"] = df["mag_auto_r_dered"] - df["mag_auto_i_dered"]
df["imz"] = df["mag_auto_i_dered"] - df["mag_auto_z_dered"]
df["zmy"] = df["mag_auto_z_dered"] - df["mag_auto_y_dered"]
df = df[FEATURE_COLUMNS + ["gmr", "rmi", "imz", "zmy"]] if FEATURE_COLUMNS else df
if FEATURE_COLUMNS:
    X = df[FEATURE_COLUMNS + ["gmr", "rmi", "imz", "zmy"]].values
else:
    X = df.values
splits = np.array_split(df, N_CLIENTS)
for i, split in enumerate(splits):
    split.to_csv(f"client_{i}.csv", index=False)
print(f"Dataset split into {N_CLIENTS} parts.")

print("Running local KMeans...")
if "coadd_object_id" in df.columns:
    X = df.drop(columns=["coadd_object_id"]).values
kmeans_local = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(X)
labels_local = kmeans_local.labels_
centers_local = kmeans_local.cluster_centers_

#save to txt
np.savetxt("local_kmeans_centers.txt", centers_local)


plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels_local, palette="tab10", s=20)
plt.title("Local KMeans Clusters (first 2 features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("local_kmeans_clusters.png")
plt.show()

print("Local KMeans inertia:", kmeans_local.inertia_)

print("Cluster centers (local):")
print(centers_local)

print("Done.")
