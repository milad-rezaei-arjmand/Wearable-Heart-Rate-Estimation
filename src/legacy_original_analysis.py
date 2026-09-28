# ---------- BEFORE ANY TF IMPORTS ----------
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"  # جلوگیری از کامپایل‌های کند XLA

import time
from time import perf_counter
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold  # در صورت نیاز: from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from scipy.signal import butter, filtfilt, lfilter  # lfilter برای حالت causal

import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
# از EarlyStopping در صورت تمایل می‌توانید استفاده کنید:
# from tensorflow.keras.callbacks import EarlyStopping

# ========== GPU setup ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    # mixed precision اختیاری؛ اگر خواستی سرعت بیشتر:
    # from tensorflow.keras import mixed_precision
    # mixed_precision.set_global_policy('mixed_float16')
    print(f"✅ GPU detected: {gpus[0].name}")
else:
    print("⚠️ No GPU detected. Using CPU.")

# ========== Filters ==========
def kalman_filter(signal, Q=0.0001, R=0.01):
    """
    فیلتر کالمن 1بعدی برای یک سگمنت مستقل.
    توجه: این تابع باید فقط روی سگمنت Train یا Test به‌طور جداگانه صدا زده شود.
    """
    n = len(signal)
    if n == 0:
        return signal.astype(np.float32)
    x_est = np.zeros(n, dtype=np.float32)
    P = np.zeros(n, dtype=np.float32)
    x_est[0] = signal[0]
    P[0] = 1.0
    for k in range(1, n):
        x_pred = x_est[k - 1]
        P_pred = P[k - 1] + Q
        K = P_pred / (P_pred + R)
        x_est[k] = x_pred + K * (signal[k] - x_pred)
        P[k] = (1 - K) * P_pred
    return x_est

def butterworth_filter_segment(signal, order=3, cutoff=0.1, zero_phase=True):
    """
    فیلتر Butterworth روی یک سگمنت مستقل بدون نشت به سگمنت دیگر.
    اگر طول سگمنت برای filtfilt کافی نباشد یا zero_phase=False باشد از lfilter (causal) استفاده می‌شود.
    """
    b, a = butter(order, cutoff, btype='low', analog=False)
    signal = np.asarray(signal, dtype=np.float32)
    if not zero_phase:
        return lfilter(b, a, signal).astype(np.float32)

    # filtfilt نیاز به padlen دارد؛ اگر کوتاه بود به lfilter برمی‌گردیم
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(signal) > padlen:
        try:
            return filtfilt(b, a, signal).astype(np.float32)
        except ValueError:
            return lfilter(b, a, signal).astype(np.float32)
    else:
        return lfilter(b, a, signal).astype(np.float32)

# ========== DNN (build once, reuse by resetting weights) ==========
def build_dnn():
    model = Sequential([
        Input(shape=(1,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, dtype='float32')  # اگر mixed precision را فعال کردی، خروجی را float32 نگه دار
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    return model

base_model = build_dnn()
base_weights = base_model.get_weights()
def fresh_model():
    m = clone_model(base_model)
    m.compile(optimizer=Adam(0.01), loss='mse')
    m.set_weights(base_weights)
    return m

# ========== Files ==========
files = {
    "Apple Watch": "Apple watch.csv",
    "Biovotion": "Biovotion.csv",
    "Empatica": "Empatica.csv",
    "Fitbit": "Fitbit.csv",
    "Garmin": "Garmin.csv",
    "Miband": "Miband.csv"
}

# ========== Merge ==========
merged_df = None
for device_name, path in files.items():
    df_i = pd.read_csv(path)
    df_i = df_i[['ECG', 'Activity', 'ID', device_name]]
    merged_df = df_i if merged_df is None else pd.merge(
        merged_df, df_i, on=['ECG', 'Activity', 'ID'], how='outer'
    )

df = merged_df.copy()
df['Activity'] = df['Activity'].fillna("Unknown")
df = df[df['Activity'] != "Unknown"]  # حذف Unknown طبق خواسته‌ی شما

devices = list(files.keys())
activities = df['Activity'].dropna().unique().tolist()
filter_types = ['Raw', 'Kalman', 'Butterworth']

# ========== Helpers for progress ==========
def count_total_tasks():
    total = 0
    for ftype in filter_types:
        for device in devices:
            for act in activities:
                if device not in df.columns:
                    continue
                temp = df[df['Activity'] == act][['ECG', device]].dropna()
                if len(temp) >= 10:
                    total += 1
    return total

TOTAL_TASKS = count_total_tasks()
done_tasks = 0

# ========== CV settings ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # در صورت سری زمانی، بهتر است از TimeSeriesSplit استفاده شود
# اگر بخواهید از نشت بین سوژه‌ها جلوگیری کنید:
# gkf = GroupKFold(n_splits=5)  # بعداً به جای kf.split(X_raw) از gkf.split(X_raw, y, groups=temp['ID'].values) استفاده کنید

EPOCHS = 120
BATCH  = 128
PATIENCE = 10  # اگر EarlyStopping خواستید، فعالش کنید

results = []

t0_all = perf_counter()
print(f"\n🚀 Starting… Total combos to run: {TOTAL_TASKS}\n")

for ftype in filter_types:
    t0_filter = perf_counter()
    print(f"\n================= FILTER: {ftype} =================")
    for device in devices:
        for act in activities:
            if device not in df.columns:
                continue

            temp = df[df['Activity'] == act][['ECG', 'ID', device]].dropna()
            n = len(temp)
            if n < 10:
                continue

            # خام (بدون فیلتر) — فقط برای split استفاده می‌شود
            X_raw_all = temp[device].values.astype(np.float32).reshape(-1, 1)
            y_all     = temp['ECG'].values.astype(np.float32)
            # groups = temp['ID'].values  # اگر GroupKFold خواستید

            done_tasks += 1
            t0_combo = perf_counter()
            print(f"[{done_tasks}/{TOTAL_TASKS}] {ftype} | {device} | {act} | n={n}")

            # storage for 5 folds
            mae_lin, rmse_lin, r2_lin = [], [], []
            mae_poly, rmse_poly, r2_poly = [], [], []
            mae_dnn, rmse_dnn, r2_dnn = [], [], []

            fold_id = 0
            # برای GroupKFold:
            # splitter = gkf.split(X_raw_all, y_all, groups=groups)
            # برای KFold ساده:
            splitter = kf.split(X_raw_all)

            for train_idx, test_idx in splitter:
                fold_id += 1

                # X خام مخصوص هر فولد
                X_train_raw = X_raw_all[train_idx].ravel()
                X_test_raw  = X_raw_all[test_idx].ravel()
                y_train     = y_all[train_idx]
                y_test      = y_all[test_idx]

                # --- اعمال فیلتر فقط روی سگمنت‌های Train/Test ---
                if ftype == 'Raw':
                    X_train = X_train_raw
                    X_test  = X_test_raw
                elif ftype == 'Kalman':
                    X_train = kalman_filter(X_train_raw)
                    X_test  = kalman_filter(X_test_raw)
                elif ftype == 'Butterworth':
                    # zero_phase=True یعنی filtfilt روی هر سگمنت؛ اگر آنلاین می‌خواهید، صفرش کنید تا lfilter شود
                    X_train = butterworth_filter_segment(X_train_raw, order=3, cutoff=0.1, zero_phase=True)
                    X_test  = butterworth_filter_segment(X_test_raw,  order=3, cutoff=0.1, zero_phase=True)
                else:
                    raise ValueError("Unknown filter type")

                # تبدیل به 2D برای مدل‌های اسکیکت‌لِرن
                X_train = X_train.reshape(-1, 1).astype(np.float32)
                X_test  = X_test.reshape(-1, 1).astype(np.float32)

                # ----- Linear -----
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                pred_lr = lr.predict(X_test)
                mae_lin.append(mean_absolute_error(y_test, pred_lr))
                rmse_lin.append(np.sqrt(np.mean((y_test - pred_lr) ** 2)))
                r2_lin.append(r2_score(y_test, pred_lr))

                # ----- Polynomial (deg=3) -----
                poly = PolynomialFeatures(degree=3)
                Xtr_poly = poly.fit_transform(X_train)
                Xte_poly = poly.transform(X_test)
                lr_poly = LinearRegression()
                lr_poly.fit(Xtr_poly, y_train)
                pred_poly = lr_poly.predict(Xte_poly)
                mae_poly.append(mean_absolute_error(y_test, pred_poly))
                rmse_poly.append(np.sqrt(np.mean((y_test - pred_poly) ** 2)))
                r2_poly.append(r2_score(y_test, pred_poly))

                # ----- DNN -----
                scaler = StandardScaler().fit(X_train)
                Xtr = scaler.transform(X_train)
                Xte = scaler.transform(X_test)

                model = fresh_model()
                # اگر EarlyStopping می‌خواهید، این دو خط را از کامنت خارج کنید:
                # from tensorflow.keras.callbacks import EarlyStopping
                # es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
                model.fit(Xtr, y_train, epochs=EPOCHS, batch_size=BATCH, verbose=0)  # , callbacks=[es]
                pred_dnn = model.predict(Xte, verbose=0).ravel()
                mae_dnn.append(mean_absolute_error(y_test, pred_dnn))
                rmse_dnn.append(np.sqrt(np.mean((y_test - pred_dnn) ** 2)))
                r2_dnn.append(r2_score(y_test, pred_dnn))

                print(f"  • Fold {fold_id}/5 done "
                      f"| Linear MAE={mae_lin[-1]:.4f} | Poly MAE={mae_poly[-1]:.4f} | DNN MAE={mae_dnn[-1]:.4f}")

            # summarize & print per combo
            def summarize(name, mae_list, rmse_list, r2_list):
                mae_arr = np.array(mae_list, dtype=float)
                rmse_arr = np.array(rmse_list, dtype=float)
                r2_arr = np.array(r2_list, dtype=float)
                out = {
                    'Device': device, 'Activity': act, 'Filter': ftype, 'Model': name,
                    'MAE_mean': float(mae_arr.mean()), 'MAE_std': float(mae_arr.std(ddof=1)),
                    'RMSE_mean': float(rmse_arr.mean()), 'RMSE_std': float(rmse_arr.std(ddof=1)),
                    'R2_mean': float(r2_arr.mean()), 'R2_std': float(r2_arr.std(ddof=1)),
                }
                return out

            r_lin  = summarize('Linear',     mae_lin,  rmse_lin,  r2_lin)
            r_poly = summarize('Polynomial', mae_poly, rmse_poly, r2_poly)
            r_dnn  = summarize('Deep',       mae_dnn,  rmse_dnn,  r2_dnn)
            results.extend([r_lin, r_poly, r_dnn])

            print(f"  → Summary {device}|{act}|{ftype}: "
                  f"Linear MAE={r_lin['MAE_mean']:.4f}±{r_lin['MAE_std']:.4f} | "
                  f"Poly MAE={r_poly['MAE_mean']:.4f}±{r_poly['MAE_std']:.4f} | "
                  f"DNN MAE={r_dnn['MAE_mean']:.4f}±{r_dnn['MAE_std']:.4f} "
                  f"⏱ {perf_counter()-t0_combo:.1f}s")

    print(f"⏱ Filter {ftype} took {perf_counter()-t0_filter:.1f}s")

# ========== Save CSV ==========
results_df = pd.DataFrame(results, columns=[
    'Device','Activity','Filter','Model',
    'MAE_mean','MAE_std','RMSE_mean','RMSE_std','R2_mean','R2_std'
]).sort_values(['Activity','Device','Filter','Model']).reset_index(drop=True)

out_name = "final_results_by_filter.csv"
results_df.to_csv(out_name, index=False)

print(f"\n✅ Saved: {out_name}")
print(f"⏱ TOTAL: {perf_counter()-t0_all:.1f}s")
print(results_df.head(12))
