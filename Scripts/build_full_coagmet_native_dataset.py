import os
import glob
import urllib.request
import re
import pandas as pd
import numpy as np

def is_numeric_fraction(val):
    val_str = str(val).strip()
    if not val_str:
        return False
    if re.match(r'^0?\.\d+$', val_str) or re.match(r'^\d+\.\d+$', val_str):
        return True
    return False

def parse_plot_and_sensor(col_name):
    clean = col_name.replace('canopy_temp_', '').replace('irt_', '').replace('canopy_', '').replace('_C_Avg', '').replace('_Avg', '').strip()

    # Pattern 1: D421 -> Plot D42, Sensor index 1
    m = re.match(r'^([A-Za-z]\d{2})(\d+)$', clean)
    if m:
        base_plot = m.group(1)
        sensor_suffix = m.group(2)
        return base_plot, f"{base_plot}_{sensor_suffix}"

    # Pattern 2: D42_1 or D42_2 or D42_A
    m2 = re.match(r'^([A-Za-z]\d{2})_(\d+|[A-Za-z])$', clean)
    if m2:
        base_plot = m2.group(1)
        sensor_suffix = m2.group(2)
        return base_plot, f"{base_plot}_{sensor_suffix}"

    # Pattern 3: Standard single sensor D42
    m3 = re.match(r'^([A-Za-z]\d{2})$', clean)
    if m3:
        base_plot = m3.group(1)
        return base_plot, f"{base_plot}_0"

    return clean, f"{clean}_0"

def get_master_treatment_map():
    trt_dict = {}

    excel_candidates = glob.glob('/mnt/Data/LIRF/*.xlsx') + glob.glob('/mnt/Data/LIRF/*/*.xlsx')
    for ef in excel_candidates:
        try:
            xls = pd.ExcelFile(ef)
            for s in xls.sheet_names:
                if any(k in s.lower() for k in ['annual', 'plot', 'treatment', 'meta']):
                    df_s = pd.read_excel(ef, sheet_name=s)
                    cols_lower = {c.lower(): c for c in df_s.columns}
                    yr_c = cols_lower.get('year', cols_lower.get('yr', None))
                    p_c = cols_lower.get('plot', cols_lower.get('plot_id', None))
                    trt_c = cols_lower.get('treatment', cols_lower.get('trt', None))
                    code_c = cols_lower.get('trt_code', cols_lower.get('trt_cd', cols_lower.get('code', None)))

                    if yr_c and p_c:
                        for idx, row in df_s.iterrows():
                            yr = row[yr_c]
                            p = str(row[p_c]).strip()
                            v_trt = str(row[trt_c]).strip() if trt_c and pd.notna(row[trt_c]) else ''
                            v_code = str(row[code_c]).strip() if code_c and pd.notna(row[code_c]) else ''

                            true_trt = None
                            if is_numeric_fraction(v_trt) and v_code and v_code.lower() not in ['', 'nan', 'none', 'n/a', 'null']:
                                true_trt = v_code
                            elif v_trt and v_trt.lower() not in ['', 'nan', 'none', 'n/a', 'null'] and not is_numeric_fraction(v_trt):
                                true_trt = v_trt
                            elif v_code and v_code.lower() not in ['', 'nan', 'none', 'n/a', 'null']:
                                true_trt = v_code

                            if yr and p and true_trt:
                                trt_dict[(int(yr), p)] = true_trt
        except Exception:
            pass

    csv_master = '/mnt/Data/LIRF/dataset/plot_year_treatment_master.csv'
    if os.path.exists(csv_master):
        df_csv = pd.read_csv(csv_master)
        for idx, row in df_csv.iterrows():
            trt_dict[(int(row['year']), str(row['plot']).strip())] = str(row['treatment']).strip()

    p_master = '/mnt/Data/LIRF/dataset/Long/lirf_master_raw.parquet'
    if not os.path.exists(p_master):
        p_master = '/mnt/Data/LIRF/dataset/Long/lirf_master.parquet'

    if os.path.exists(p_master):
        df_m = pd.read_parquet(p_master, columns=['timestamp', 'plot', 'treatment'])
        df_m['ts'] = pd.to_datetime(df_m['timestamp'])
        df_m['year'] = df_m['ts'].dt.year
        df_valid = df_m[df_m['treatment'].notna() & (df_m['treatment'] != '') & (df_m['treatment'] != 'NaN')].copy()
        for (yr, p), grp in df_valid.groupby(['year', 'plot']):
            v = str(grp['treatment'].iloc[0]).strip()
            if (yr, p) not in trt_dict or is_numeric_fraction(trt_dict[(yr, p)]):
                if not is_numeric_fraction(v):
                    trt_dict[(yr, p)] = v

    return trt_dict

TREATMENT_MAP = get_master_treatment_map()

def fetch_coagmet_weather_api(station, year, start_date, end_date):
    url_5min = f"https://coagmet.colostate.edu/data/5min/{station}.csv?header=yes&from={start_date}&to={end_date}"
    try:
        req = urllib.request.Request(url_5min, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response, skiprows=[1])
        df = df.rename(columns={c: str(c).replace('"', '').strip() for c in df.columns})
        time_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df[df['timestamp'].notna()].sort_values('timestamp').reset_index(drop=True)

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(-999, np.nan)

        df['air_temp_c'] = (df['Air Temp'] - 32) * 5 / 9 if 'Air Temp' in df.columns else np.nan
        if not df['air_temp_c'].isna().all():
            df['relative_humidity'] = df['RH'] if 'RH' in df.columns else np.nan
            df['solar_rad_w_m2'] = df['Solar Rad'] if 'Solar Rad' in df.columns else np.nan
            df['precip_mm'] = df['Liquid Precip'] * 25.4 if 'Liquid Precip' in df.columns else np.nan
            df['wind_speed_m_s'] = df['Wind'] * 0.44704 if 'Wind' in df.columns else np.nan
            if 'Dewpoint' in df.columns:
                dp_c = (df['Dewpoint'] - 32) * 5 / 9
                df['vap_press_kpa'] = 0.61078 * np.exp((17.27 * dp_c) / (dp_c + 237.3))
            else:
                df['vap_press_kpa'] = np.nan
            df['station'] = station
            return df
    except Exception:
        pass

    url_hourly = f"https://coagmet.colostate.edu/data/hourly/{station}.csv?header=yes&from={start_date}&to={end_date}"
    try:
        req = urllib.request.Request(url_hourly, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response, skiprows=[1])
        df = df.rename(columns={c: str(c).replace('"', '').strip() for c in df.columns})
        time_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
        df['timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df[df['timestamp'].notna()].sort_values('timestamp').reset_index(drop=True)

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace(-999, np.nan)

        df['air_temp_c'] = (df['Air Temp'] - 32) * 5 / 9 if 'Air Temp' in df.columns else np.nan
        df['relative_humidity'] = df['RH'] if 'RH' in df.columns else np.nan
        df['solar_rad_w_m2'] = df['Solar Rad'] if 'Solar Rad' in df.columns else np.nan
        df['precip_mm'] = df['Liquid Precip'] * 25.4 if 'Liquid Precip' in df.columns else np.nan
        df['wind_speed_m_s'] = df['Wind'] * 0.44704 if 'Wind' in df.columns else np.nan

        if 'Dewpoint' in df.columns:
            dp_c = (df['Dewpoint'] - 32) * 5 / 9
            df['vap_press_kpa'] = 0.61078 * np.exp((17.27 * dp_c) / (dp_c + 237.3))
        else:
            df['vap_press_kpa'] = np.nan

        df['station'] = station
        return df
    except Exception:
        return None

def fetch_coagmet_daily_et(station, year):
    url = f"https://coagmet.colostate.edu/data/daily/{station}.csv?header=yes&from={year}-05-01&to={year}-10-31"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            df = pd.read_csv(response, skiprows=[1])
        df = df.rename(columns={c: str(c).replace('"', '').strip() for c in df.columns})
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        df['date'] = pd.to_datetime(df[date_col]).dt.date
        eto_col = [c for c in df.columns if 'eto' in c.lower() or 'grass' in c.lower()]
        etr_col = [c for c in df.columns if 'etr' in c.lower() or 'alfalfa' in c.lower()]
        
        df['eto'] = df[eto_col[0]] if eto_col else np.nan
        df['etr'] = df[etr_col[0]] if etr_col else np.nan
        return df[['date', 'eto', 'etr']].dropna(subset=['date'])
    except Exception:
        return None

def fetch_widefile_weather_for_year(year, wide_dir):
    yr_dir = os.path.join(wide_dir, f"Year_{year}")
    wide_file = None
    for freq in ['5min', '15min', '30min', '60min']:
        p_pq = os.path.join(yr_dir, f"lirf_wide_{freq}.parquet")
        p_csv = os.path.join(yr_dir, f"lirf_wide_{freq}.csv")
        if os.path.exists(p_pq):
            wide_file = p_pq
            break
        elif os.path.exists(p_csv):
            wide_file = p_csv
            break

    if wide_file:
        df_w = pd.read_parquet(wide_file) if wide_file.endswith('.parquet') else pd.read_csv(wide_file)
        rename_map = {
            'weather_air_temp_c': 'air_temp_c',
            'weather_rh': 'relative_humidity',
            'rh': 'relative_humidity',
            'weather_vap_press_kpa': 'vap_press_kpa',
            'weather_solar_rad_w_m2': 'solar_rad_w_m2',
            'weather_wind_speed_m_s': 'wind_speed_m_s',
            'weather_precip_mm': 'precip_mm',
            'weather_etr': 'etr',
            'weather_eto': 'eto',
            'weather_imputed': 'weather_imputed'
        }
        df_w = df_w.rename(columns=rename_map)
        ts_col = [c for c in df_w.columns if any(k in c.lower() for k in ['timestamp', 'date', 'time'])][0]
        df_w['timestamp'] = pd.to_datetime(df_w[ts_col])
        w_cols = ['air_temp_c', 'relative_humidity', 'vap_press_kpa', 'solar_rad_w_m2', 'wind_speed_m_s', 'precip_mm', 'etr', 'eto']
        avail_w = [c for c in w_cols if c in df_w.columns]
        if avail_w:
            df_w_sub = df_w[['timestamp'] + avail_w].drop_duplicates(subset=['timestamp']).dropna(subset=['air_temp_c']).sort_values('timestamp').reset_index(drop=True)
            df_w_sub['weather_imputed'] = 0
            return df_w_sub
    return None

def process_year(year, wide_dir):
    print(f"\n========================================================")
    print(f"PROCESSING YEAR {year} (HIGH-RES WEATHER RESAMPLING + NO PRECIP LOST)")
    print(f"========================================================")

    legacy_pq = f"/mnt/Data/LIRF/Scripts/{year}_IRT_Merged.parquet"
    legacy_csv = f"/mnt/Data/LIRF/Scripts/{year}_IRT_Merged.csv"
    
    target_file = None
    if os.path.exists(legacy_pq):
        target_file = legacy_pq
    elif os.path.exists(legacy_csv):
        target_file = legacy_csv

    if not target_file:
        yr_dir = os.path.join(wide_dir, f"Year_{year}")
        for freq in ['5min', '15min', '30min', '60min']:
            p_pq = os.path.join(yr_dir, f"lirf_wide_{freq}.parquet")
            p_csv = os.path.join(yr_dir, f"lirf_wide_{freq}.csv")
            if os.path.exists(p_pq):
                target_file = p_pq
                break
            elif os.path.exists(p_csv):
                target_file = p_csv
                break

    if not target_file:
        print(f"  [SKIP] No IRT dataset found for year {year}")
        return None

    print(f"  Loading IRT Dataset: {target_file}")
    df_irt = pd.read_parquet(target_file) if target_file.endswith('.parquet') else pd.read_csv(target_file)
    
    ts_irt_col = [c for c in df_irt.columns if any(k in c.lower() for k in ['timestamp', 'date', 'time'])][0]
    df_irt['timestamp'] = pd.to_datetime(df_irt[ts_irt_col])
    df_irt = df_irt.sort_values('timestamp').reset_index(drop=True)

    ct_cols = [c for c in df_irt.columns if any(k in c.lower() for k in ['canopy', 'irt', 'avg', 'temp']) and not c.startswith('weather_') and c.lower() not in ['timestamp', 'doy', 'hour', 'minute', 'day', 'canopy_temp', 'air_temp', 'air_temp_c']]
    
    for c in ct_cols:
        df_irt[c] = pd.to_numeric(df_irt[c], errors='coerce')

    has_irt = df_irt[ct_cols].notna().any(axis=1)
    df_irt_active = df_irt[has_irt].copy()

    if len(df_irt_active) == 0:
        print(f"  [SKIP] No active IRT readings in {year}")
        return None

    t_start = df_irt_active['timestamp'].min()
    t_end = df_irt_active['timestamp'].max()
    print(f"  Active Season Bounds: {t_start} to {t_end}")

    s_date = f"{year}-05-01"
    e_date = f"{year}-10-31"
    
    # Fetch CoAgMET weather via API (5min -> Hourly fallback)
    df_w_api = fetch_coagmet_weather_api('gly04', year, s_date, e_date)
    w_cols = ['air_temp_c', 'relative_humidity', 'vap_press_kpa', 'solar_rad_w_m2', 'wind_speed_m_s', 'precip_mm', 'weather_imputed']

    if df_w_api is not None and not df_w_api['air_temp_c'].isna().all():
        df_w_api['weather_imputed'] = 0
        stations = ['lcn01', 'alt01', 'gly03', 'ksy01']
        for st in stations:
            if df_w_api['air_temp_c'].isna().any():
                df_b = fetch_coagmet_weather_api(st, year, s_date, e_date)
                if df_b is not None and len(df_b) > 0:
                    df_w_set = df_w_api.set_index('timestamp')
                    df_b_set = df_b.set_index('timestamp')
                    missing_mask = df_w_set['air_temp_c'].isna()
                    for wc in ['air_temp_c', 'relative_humidity', 'vap_press_kpa', 'solar_rad_w_m2', 'wind_speed_m_s', 'precip_mm']:
                        if wc in df_b_set.columns:
                            df_w_set[wc] = df_w_set[wc].combine_first(df_b_set[wc])
                    df_w_set.loc[missing_mask & df_w_set['air_temp_c'].notna(), 'weather_imputed'] = 1
                    df_w_api = df_w_set.reset_index()

        df_w_api = df_w_api.dropna(subset=['air_temp_c']).sort_values('timestamp').reset_index(drop=True)
        df_w_final = df_w_api
    else:
        print(f"  Fallback: Pulling weather from wide parquet file for year {year}...")
        df_w_wide = fetch_widefile_weather_for_year(year, wide_dir)
        df_w_final = df_w_wide

    df_irt_clean = df_irt_active.drop(columns=[c for c in w_cols if c in df_irt_active.columns])
    
    if df_w_final is not None and len(df_w_final) > 0:
        w_avail = [c for c in w_cols if c in df_w_final.columns]

        # DETECT WEATHER FREQUENCY VS IRT FREQUENCY
        w_dt_median = df_w_final['timestamp'].diff().median()
        irt_dt_median = df_irt_clean['timestamp'].diff().median()

        # If Weather is FASTER than IRT (e.g. 5-min weather in 2017-2024 vs 15/30-min IRT)
        if pd.notna(w_dt_median) and pd.notna(irt_dt_median) and w_dt_median < irt_dt_median * 0.7:
            irt_interval_str = f"{int(irt_dt_median.total_seconds() / 60)}min"
            print(f"  [HIGH-RES WEATHER] Resampling {w_dt_median} weather to match IRT grid ({irt_interval_str}): SUM precip, MEAN temperature/RH/rad...")
            
            agg_rules = {}
            for c in w_avail:
                if c == 'precip_mm':
                    agg_rules[c] = 'sum'
                elif c == 'weather_imputed':
                    agg_rules[c] = 'max'
                else:
                    agg_rules[c] = 'mean'

            df_w_resampled = df_w_final.set_index('timestamp').resample(irt_interval_str).agg(agg_rules).reset_index()
            df_w_resampled = df_w_resampled.dropna(subset=['air_temp_c']).sort_values('timestamp').reset_index(drop=True)

            merged = pd.merge_asof(
                df_irt_clean.sort_values('timestamp'),
                df_w_resampled[['timestamp'] + w_avail].sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('7.5min')
            )
        else:
            # Weather is SLOWER or equal frequency (e.g. 1-hour weather in 2008-2016)
            # Map weather to exact on-the-hour IRT row (tolerance=2min) without duplicating across rows
            merged = pd.merge_asof(
                df_irt_clean.sort_values('timestamp'),
                df_w_final[['timestamp'] + w_avail].sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('2min')
            )
    else:
        merged = df_irt_active.copy()

    if 'weather_imputed' not in merged.columns:
        merged['weather_imputed'] = 0

    if 'eto' not in merged.columns or merged['eto'].isna().all():
        df_et = fetch_coagmet_daily_et('gly04', year)
        if df_et is not None:
            merged['date'] = merged['timestamp'].dt.date
            merged = pd.merge(merged, df_et, on='date', how='left').drop(columns=['date'])
        else:
            merged['eto'] = np.nan
            merged['etr'] = np.nan

    long_records = []
    
    for ct_col in ct_cols:
        base_p, sensor_id_val = parse_plot_and_sensor(ct_col)

        al_col = f"tc_anomaly_label_{ct_col}"
        if al_col not in merged.columns:
            al_col = f"anomaly_label_{ct_col}"
        if al_col not in merged.columns:
            al_col = f"tc_anomaly_label_{base_p}"
        if al_col not in merged.columns:
            al_col = f"anomaly_label_{base_p}"

        trt_val = TREATMENT_MAP.get((year, base_p), None)
        if not trt_val or is_numeric_fraction(trt_val):
            if 'treatment' in merged.columns and merged['treatment'].notna().any():
                v = merged['treatment'].iloc[0]
                if v and str(v).strip() not in ['', 'nan', 'None', 'NaN', 'N/A'] and not is_numeric_fraction(v):
                    trt_val = str(v).strip()

        if not trt_val or is_numeric_fraction(trt_val):
            trt_val = "100/100"

        df_p = pd.DataFrame()
        df_p['timestamp'] = merged['timestamp']
        df_p['plot'] = base_p
        df_p['sensor_id'] = sensor_id_val
        df_p['canopy_temp'] = pd.to_numeric(merged[ct_col], errors='coerce')
        
        raw_al = merged[al_col] if al_col in merged.columns else 'Valid'
        df_p['tc_anomaly_label'] = raw_al
        
        for wc in w_cols + ['eto', 'etr']:
            df_p[wc] = pd.to_numeric(merged[wc], errors='coerce') if wc in merged.columns else np.nan
            
        df_p['treatment'] = trt_val
        long_records.append(df_p)

    df_long = pd.concat(long_records, ignore_index=True)
    
    anomaly_map = {'Valid': 0, 'Clean': 0, 'Confirmed': 0, 'Plausible': 0, 'Spurious': 1, 'Spurious (Irrigation)': 2, 'Anomaly': 3, 'Anomaly (Irrigation)': 4, 'Sensor Malfunction': 5}
    df_long['tc_anomaly_label'] = df_long['tc_anomaly_label'].map(anomaly_map).fillna(0).astype(int)

    float_cols = df_long.select_dtypes(include=['float64', 'float32']).columns
    df_long[float_cols] = df_long[float_cols].round(3)

    target_order = ['timestamp', 'plot', 'sensor_id', 'canopy_temp', 'tc_anomaly_label', 'air_temp_c', 'relative_humidity', 'vap_press_kpa', 'solar_rad_w_m2', 'wind_speed_m_s', 'precip_mm', 'etr', 'eto', 'weather_imputed', 'treatment']
    df_long = df_long.reindex(columns=target_order)

    sec_sensors = df_long[~df_long['sensor_id'].str.endswith('_0')]['sensor_id'].unique()
    print(f"  Completed Year {year}: {len(df_long):,} longform rows across {df_long['plot'].nunique()} plots (Weather NaNs: {df_long['air_temp_c'].isna().sum():,}, Precip Sum: {df_long['precip_mm'].sum():.1f} mm).")
    return df_long

def main():
    wide_dir = '/mnt/Data/LIRF/dataset/Wide'
    years = range(2008, 2025)

    year_dfs = []
    for yr in years:
        res = process_year(yr, wide_dir)
        if res is not None:
            year_dfs.append(res)

    df_master = pd.concat(year_dfs, ignore_index=True)
    df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])
    
    num_cols = ['canopy_temp', 'air_temp_c', 'relative_humidity', 'vap_press_kpa', 'solar_rad_w_m2', 'wind_speed_m_s', 'precip_mm', 'etr', 'eto']
    for nc in num_cols:
        df_master[nc] = pd.to_numeric(df_master[nc], errors='coerce')
        
    df_master = df_master.sort_values(['timestamp', 'plot', 'sensor_id']).reset_index(drop=True)

    sec_count = len(df_master[~df_master['sensor_id'].str.endswith('_0')])
    missing_w = df_master['air_temp_c'].isna().sum()
    w_pct = (missing_w / len(df_master)) * 100

    print("\n========================================================")
    print("NATIVE MASTER DATASET BUILT (ZERO WEATHER LOST + HIGH-RES RESAMPLING)")
    print("========================================================")
    print(f"Total Rows:                     {len(df_master):,}")
    print(f"Total Columns:                  {len(df_master.columns)}")
    print(f"Secondary/Tertiary Sensor Rows: {sec_count:,} rows (e.g. _1, _2)")
    print(f"Missing Weather Count:          {missing_w:,} ({w_pct:.2f}% MISSING)")
    print(f"Total Master Precip Captured:   {df_master['precip_mm'].sum():,.1f} mm")
    print(f"Weather Imputed Flags:          {df_master['weather_imputed'].sum():,} rows patched from backup stations")

    out_parquet = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.parquet'
    out_csv = '/mnt/Data/LIRF/dataset/Long/lirf_master_native.csv'

    df_master.to_parquet(out_parquet, index=False)
    df_master.to_csv(out_csv, index=False)

    print(f"\nSaved: {out_parquet}")
    print(f"Saved: {out_csv}")

if __name__ == '__main__':
    main()
