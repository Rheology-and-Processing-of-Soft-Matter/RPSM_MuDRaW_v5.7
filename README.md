# Using GUI

Will add...

## Viscosity Data Processing

This script processes steady shear viscosity data exported from an Anton Paar rheometer. It is intended for use with viscosity measurements performed using a custom 2-interval procedure (1st interval to reach steady state, 2nd interval to measure viscosity). The script identifies steady-state intervals, calculates average values for viscosity, shear rate, and shear stress, and saves the results in a structured format for further analysis.

---

### Input Data

- Tab-separated `.txt` file exported from the rheometer.
- The first 11 rows are skipped automatically.
- Expected columns (headers are assigned in the code):

"Interval Data", "Point No.", "Time", "Shear Rate", "Viscosity", "Shear Stress", "Torque", "Gap", "Normal Force", "Temperature", "Test Start Date and Time", "Time of Day", "Status"
<img width="878" alt="image" src="https://github.com/user-attachments/assets/6898f5bc-dc6c-41c0-b229-d4edb2893e0a" />

---

### Output Data

Processed files are saved in a subfolder named `_Processed`:

- `1_<sample_name>_shear_rate.csv`  
  Averaged shear rate values (1/s)

- `2_<sample_name>_steady_state_shear_stress.csv`  
  Averaged shear stress values (Pa)

- `3_<sample_name>_steady_state_viscosity.csv`  
  Averaged viscosity values (Pa·s)

- `<sample_name>_indices.csv`  
  Indices marking the start of each steady-state segment


### Interval Averaging

The script uses two parameters — `1st point to average` and `Enter how many points to average` — to define which part of interval (where viscosity is measured) is used for averaging. These are passed as command-line arguments when running the script.

<img width="371" alt="image" src="https://github.com/user-attachments/assets/05be6a2a-0c87-4eb1-afdc-d15caea3cefe" />

- `1st point to average`: number of data points to skip at the start of each measurement interval (to avoid transients).
- `Enter how many points to average`: number of data points to include after the skipped portion for averaging.

This allows flexible selection of the most stable portion of each steady shear interval for analysis.


