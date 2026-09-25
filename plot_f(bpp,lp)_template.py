import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io

# ==========================================
# EDITABLE BLOCK 1: Mathematical Operation
# ==========================================
ALPHA_VAL = 2.0
ALPHA_ERR = 0.2

def function1(bpp, lp):
# edit to desired definition of f(bpp, lp)
    return (bpp-lp)/ALPHA_VAL

function_name = r"$T_e$ [eV]" #edit to desired variable


# ==========================================
# EDITABLE BLOCK 2: GOLEM Web Endpoint Templates
# ==========================================
# Default URL endpoints for probe diagnostics on the GOLEM server
# Adjust sub-paths if your specific diagnostic files are named/located differently
BPP_URL_TEMPLATE = "http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/DoubleLangBallPenProbe/BPP_HFS.csv"
LP_URL_TEMPLATE = "http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/DoubleLangBallPenProbe/LP_left.csv"
# Adjust left/right and HFS/LFS correspondingly

# ==========================================
# EDITABLE BLOCK 3: Plasma Shift Thresholds
# ==========================================
# Discard any data where the absolute shift |dz| exceeds this limit (in mm); omit start and end of discharges
DZ_MAX_THRESHOLD_MM = 30.0 

# Units multiplier for camera shift (1000.0 if meters, 1.0 if mm)
DZ_UNIT_MULTIPLIER = 1.0 


# ==========================================
# EDITABLE BLOCK 4: Custom Time Intervals
# ==========================================
custom_intervals = [
    (2.5, 3.5),
    (3.5, 4.5),
    (4.5, 5.5)
]

def fetch_golem_csv(url, col_names=None):
    """
    Helper function to fetch and parse space/comma-delimited CSVs directly from GOLEM.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    if col_names:
        df = pd.read_csv(io.StringIO(response.text), sep=r'[,\s]+', engine='python', header=None, names=col_names)
    else:
        df = pd.read_csv(io.StringIO(response.text), sep=r'[,\s]+', engine='python')
    return df


def process_and_plot_dynamic_rho(shots_and_radii, time_limit=12.0, bucket_size=0.25):
    all_averaged_data = []

    # ==========================================
    # STEP 1: Process, Fetch Data, Merge, and Bucket
    # ==========================================
    for shot, r_nominal in shots_and_radii.items():
        print(f"\n--- Processing Shot {shot} ---")
        
        # --- 1A: Fetch Probe Data Directly from GOLEM ---
        bpp_url = BPP_URL_TEMPLATE.format(shot=shot)
        lp_url = LP_URL_TEMPLATE.format(shot=shot)

        try:
            df_bpp = fetch_golem_csv(bpp_url)
            df_lp = fetch_golem_csv(lp_url)

            # Ensure numeric conversion and assign standard column names
            df_bpp.rename(columns={df_bpp.columns[0]: 't', df_bpp.columns[1]: 'bpp'}, inplace=True)
            df_lp.rename(columns={df_lp.columns[0]: 't', df_lp.columns[1]: 'lp'}, inplace=True)

            df_bpp['t'] = pd.to_numeric(df_bpp['t'], errors='coerce')
            df_bpp['bpp'] = pd.to_numeric(df_bpp['bpp'], errors='coerce')
            df_lp['t'] = pd.to_numeric(df_lp['t'], errors='coerce')
            df_lp['lp'] = pd.to_numeric(df_lp['lp'], errors='coerce')

            df_bpp.dropna(inplace=True)
            df_lp.dropna(inplace=True)

        except Exception as e:
            print(f"Warning: Could not retrieve/parse probe data for shot {shot}. Skipping. Error: {e}")
            continue

        # Oscilloscope scaling multiplier
        df_bpp['bpp'] = df_bpp['bpp'] * 100 
        df_lp['lp'] = df_lp['lp'] * 100

        # Convert time to ms if in seconds
        if df_bpp['t'].max() < 2.0:
            df_bpp['t'] *= 1000.0
            df_lp['t'] *= 1000.0

        df_bpp = df_bpp.sort_values('t')
        df_lp = df_lp.sort_values('t')


        # --- 1B: Fetch Fast Camera Data ---
        url_dz = f"http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/FastCameras/Camera_Vertical/CameraVerticalPosition"
        try:
            df_dz = fetch_golem_csv(url_dz, col_names=['t', 'dz'])
            df_dz['t'] = pd.to_numeric(df_dz['t'], errors='coerce')
            df_dz['dz'] = pd.to_numeric(df_dz['dz'], errors='coerce')
            df_dz.dropna(inplace=True)

            if df_dz.empty:
                print(f"Shot {shot}: Camera dz data is empty after parsing. Skipping.")
                continue

            if df_dz['t'].max() < 2.0:
                df_dz['t'] *= 1000.0

            df_dz['dz'] *= DZ_UNIT_MULTIPLIER
            print(f"Max Absolute Shift (|dz|): {df_dz['dz'].abs().max():.2f} mm")

            df_dz = df_dz[df_dz['dz'].abs() <= DZ_MAX_THRESHOLD_MM].sort_values('t')

            if df_dz.empty:
                print(f"Shot {shot}: All camera dz data dropped because shift exceeded {DZ_MAX_THRESHOLD_MM}mm limit!")
                continue

        except Exception as e:
            print(f"Warning: Could not retrieve vertical camera data for shot {shot}. Error: {e}")
            continue

        url_dr = f"http://golem.fjfi.cvut.cz/shots/{shot}/Diagnostics/FastCameras/Camera_Radial/CameraRadialPosition"
        try:
            df_dr = fetch_golem_csv(url_dr, col_names=['t', 'dr'])
            df_dr['t'] = pd.to_numeric(df_dr['t'], errors='coerce')
            df_dr['dr'] = pd.to_numeric(df_dr['dr'], errors='coerce')
            df_dr.dropna(inplace=True)

            if df_dr.empty:
                print(f"Shot {shot}: Camera dr data is empty after parsing. Skipping.")
                continue

            if df_dr['t'].max() < 2.0:
                df_dr['t'] *= 1000.0

            df_dr['dr'] *= DZ_UNIT_MULTIPLIER
            print(f"Max Absolute Shift (|dr|): {df_dr['dr'].abs().max():.2f} mm")

            df_dr = df_dr[df_dr['dr'].abs() <= DZ_MAX_THRESHOLD_MM].sort_values('t')

            if df_dr.empty:
                print(f"Shot {shot}: All camera dr data dropped because shift exceeded {DZ_MAX_THRESHOLD_MM}mm limit!")
                continue

        except Exception as e:
            print(f"Warning: Could not retrieve radial camera data for shot {shot}. Error: {e}")
            continue


        # --- 1C: Merge Probes and Camera Data ---
        df = pd.merge_asof(df_bpp, df_lp, on='t', direction='nearest', tolerance=0.01)
        df = pd.merge_asof(df, df_dz, on='t', direction='nearest', tolerance=0.01)
        df = pd.merge_asof(df, df_dr, on='t', direction='nearest', tolerance=0.01)

        # Prevent artificial backward shifting prior to camera trigger
        first_cam_t_z = df_dz['t'].min()
        first_cam_t_r = df_dr['t'].min()

        df.loc[df['t'] < first_cam_t_z, 'dz'] = 0.0
        df.loc[df['t'] < first_cam_t_r, 'dr'] = 0.0

        rows_before = len(df)
        df.dropna(inplace=True)
        rows_after = len(df)

        print(f"Probe Time Range : {df_bpp['t'].min():.2f} to {df_bpp['t'].max():.2f} ms")
        print(f"Camera Time Range: {df_dz['t'].min():.2f} to {df_dz['t'].max():.2f} ms")
        print(f"Rows before drop: {rows_before} | Rows after drop: {rows_after}")


        # --- 1D: Calculate Physical Variables ---
        df['rho'] = ((r_nominal + df['dz'])**2 + (df['dr'] - 3)**2)**(1/2)
        df['result'] = function1(df['bpp'], df['lp'])
        df = df[df['t'] <= time_limit]


        # --- 1E: Bucket into Time Intervals ---
        df['time_bucket'] = np.floor(df['t'] / bucket_size) * bucket_size
        grouped = df.groupby('time_bucket')

        bucketed_df = grouped.agg(
            rho_mean=('rho', 'mean'),
            result_mean=('result', 'mean'),
            result_std=('result', 'std')
        ).reset_index()

        fractional_error = ALPHA_ERR / ALPHA_VAL
        bucketed_df['result_std'] = np.sqrt(
            bucketed_df['result_std']**2 + (bucketed_df['result_mean'] * fractional_error)**2
        )

        bucketed_df['shot'] = shot
        all_averaged_data.append(bucketed_df)

    if not all_averaged_data:
        print("No valid data processed. Check shot numbers and internet connection.")
        return


    # ==========================================
    # STEP 2: Aggregate Macro-Intervals and Plot
    # ==========================================
    master_df = pd.concat(all_averaged_data)

    plt.figure(figsize=(6, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(custom_intervals)))

    for i, (start_t, end_t) in enumerate(custom_intervals):
        interval_data = master_df[(master_df['time_bucket'] >= start_t) & (master_df['time_bucket'] < end_t)]

        if interval_data.empty:
            continue

        grouped_shot = interval_data.groupby('shot')

        plot_df = grouped_shot.agg(
            x_rho=('rho_mean', 'mean'),
            y_val=('result_mean', 'mean'),
            y_err=('result_std', lambda x: np.sqrt((x**2).sum()) / len(x))
        ).reset_index()

        plot_df = plot_df.sort_values('x_rho')

        eb = plt.errorbar(
            plot_df['x_rho'], plot_df['y_val'], yerr=plot_df['y_err'],
            marker='o', linestyle='-', color=colors[i], capsize=3, capthick=1,
            label=f'{start_t} - {end_t}'
        )

        transparency_level = 0.4
        for cap in eb[1]:
            cap.set_alpha(transparency_level)
        for bar in eb[2]:
            bar.set_alpha(transparency_level)
            
        # Print formatted (rho, f(bpp, lp)) table for the current time interval
        print(f"\n==========================================")
        print(f" time interval: {start_t} - {end_t} ms")
        print(f"==========================================")
        print(f"{'rho [mm]':>8} | {function_name:>12}")
        print("-" * 42)
        for _, row in plot_df.iterrows():
            func_val_str = f"{row['y_val']:.2f} ± {row['y_err']:.2f}"
            print(f"{row['x_rho']:8.2f} | {func_val_str:>12}")


    # ==========================================
    # STEP 3: Aesthetics and Export | edit to desired title and labels
    # ==========================================
    plt.title(r'$I_p=xx$ kA', fontsize=14)
    plt.xlabel(r'$\rho$ [mm]', fontsize=14)
    plt.ylabel(function_name, fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="t [ms]")
    plt.tight_layout()

    plt.savefig('probe_analysis_generic_template.png', dpi=300)
    plt.show()


# === EXECUTION BLOCK ===
if __name__ == "__main__":
    # Map any arbitrary shot numbers to their nominal radial positions (r in mm)
    my_shots = {
        53029: 90,
        53030: 85,
        53031: 80
    }

    process_and_plot_dynamic_rho(my_shots)