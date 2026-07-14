"""
Generate a realistic crop-recommendation dataset.
Each crop has a characteristic range for N, P, K, temperature, humidity, ph, rainfall.
Values are sampled from truncated normal distributions around agronomic norms
(based on well-established agricultural extension guidelines).
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# crop: (N_mean,N_sd, P_mean,P_sd, K_mean,K_sd, temp_mean,temp_sd, hum_mean,hum_sd, ph_mean,ph_sd, rain_mean,rain_sd)
CROPS = {
    "rice":        (80, 12, 45, 10, 40, 10, 25, 2.5, 82, 5, 6.4, 0.4, 220, 30),
    "maize":       (85, 15, 40, 10, 20, 6,  24, 3.0, 62, 8, 6.2, 0.5, 90,  25),
    "chickpea":    (40, 8,  60, 10, 80, 10, 19, 3.0, 16, 5, 7.2, 0.4, 75,  20),
    "cotton":      (120,15, 45, 10, 20, 6,  27, 2.5, 70, 7, 6.9, 0.5, 85,  20),
    "coffee":      (100,12, 30, 8,  30, 8,  25, 2.0, 58, 8, 6.7, 0.5, 175, 25),
    "banana":      (100,12, 80, 10, 50, 10, 27, 2.0, 78, 6, 6.0, 0.4, 105, 20),
    "mango":       (20, 6,  27, 8,  30, 8,  32, 2.5, 50, 8, 5.9, 0.4, 95,  20),
    "grapes":      (18, 5,  130,15, 200,20, 24, 2.5, 82, 6, 6.1, 0.4, 70,  15),
    "watermelon":  (95, 12, 17, 6,  50, 10, 26, 2.0, 85, 5, 6.5, 0.4, 45,  12),
    "orange":      (18, 5,  16, 5,  10, 4,  23, 2.5, 92, 4, 6.9, 0.4, 110, 20),
    "jute":        (78, 10, 47, 10, 40, 8,  25, 2.0, 80, 5, 6.7, 0.4, 175, 25),
    "lentil":      (20, 6,  65, 10, 20, 6,  20, 3.0, 65, 8, 6.9, 0.4, 45,  12),
}

N_PER_CROP = 200
rows = []
for crop, (nm,ns, pm,ps, km,ks, tm,ts, hm,hs, phm,phs, rm,rs) in CROPS.items():
    n = np.clip(np.random.normal(nm, ns, N_PER_CROP), 0, 140)
    p = np.clip(np.random.normal(pm, ps, N_PER_CROP), 0, 145)
    k = np.clip(np.random.normal(km, ks, N_PER_CROP), 0, 205)
    temp = np.clip(np.random.normal(tm, ts, N_PER_CROP), 8, 44)
    hum = np.clip(np.random.normal(hm, hs, N_PER_CROP), 14, 100)
    ph = np.clip(np.random.normal(phm, phs, N_PER_CROP), 3.5, 9.5)
    rain = np.clip(np.random.normal(rm, rs, N_PER_CROP), 20, 300)
    for i in range(N_PER_CROP):
        rows.append([n[i], p[i], k[i], temp[i], hum[i], ph[i], rain[i], crop])

df = pd.DataFrame(rows, columns=["N","P","K","temperature","humidity","ph","rainfall","label"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.round({"N":1,"P":1,"K":1,"temperature":1,"humidity":1,"ph":2,"rainfall":1})
df.to_csv("/home/claude/crop_project/data/Data.csv", index=False)
print(df.shape)
print(df["label"].value_counts())
