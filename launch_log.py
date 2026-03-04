# data analysis for 1-18 launch log 
from os import sync
import csv
import matplotlib.pyplot as plt
import numpy as np

# Parse file 
time_data = []
xg_data = []
yg_data = []
zg_data = []
yg_high_data = []
airbrake_pct_data = []
pressure_data = []

with open('LOG028.TXT', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    reader.fieldnames = [h.strip() for h in reader.fieldnames]

    for row in reader:
        values = row['Time'].split(',')
        time_data.append(float(values[0]))
        xg_data.append(float(values[1]))
        yg_data.append(float(values[2]))
        zg_data.append(float(values[3]))
        yg_high_data.append(float(values[5]))
        airbrake_pct_data.append(float(values[18]))
        pressure_data.append(float(values[4]))

def plot_acc(time, xg, yg, zg, airbrake_pct):
    fig, axes = plt.subplots(5, 1, figsize=(16, 10))
    fig.suptitle('Acceleration Components vs Time', fontsize=14, fontweight='bold')

    # Plot Xg vs Time
    axes[0].plot(time, xg, 'r-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Xg (g)', fontsize=10)
    axes[0].set_title('Xg Acceleration vs Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot Yg vs Time
    axes[1].plot(time, yg, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Yg (g)', fontsize=10)
    axes[1].set_title('Yg Acceleration vs Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot Zg vs Time
    axes[2].plot(time, zg, 'b-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Zg (g)', fontsize=10)
    axes[2].set_xlabel('Time (seconds)', fontsize=10)
    axes[2].set_title('Zg Acceleration vs Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Plot Yg_high vs TIme
    axes[3].plot(time, yg_high_data, 'k-', linewidth=0.5, alpha=0.7)
    axes[3].set_ylabel('Yg_high (g)', fontsize=10)
    axes[3].set_xlabel('Time (seconds)', fontsize=10)
    axes[3].set_title('Yg_high Acceleration vs Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    # Plot Airbrake pct vs Time
    axes[4].plot(time, airbrake_pct, 'k-', linewidth=0.5, alpha=0.7)
    axes[4].set_ylabel('Airbrake deployment pct (%)', fontsize=10)
    axes[4].set_xlabel('Time (seconds)', fontsize=10)
    axes[4].set_title('Airbrake pct vs Time', fontsize=11)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('acceleration.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'acceleration.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} seconds")

def plot_acc_vs_ab_pct(time, yg, yg_high, airbrake_pct):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Acceleration Components vs Time', fontsize=14, fontweight='bold')

    # Plot Yg vs Time
    axes[0].plot(time, yg, 'g-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('Yg (g)', fontsize=10)
    axes[0].set_title('Yg Acceleration vs Time', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot Yg_high vs Time
    axes[1].plot(time, yg_high_data, 'g-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('Yg_high (g)', fontsize=10)
    axes[1].set_title('Yg_high Acceleration vs Time', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Plot Airbrake pct vs Time
    axes[2].plot(time, airbrake_pct, 'k-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('Airbrake deployment pct (%)', fontsize=10)
    axes[2].set_xlabel('Time (seconds)', fontsize=10)
    axes[2].set_title('Airbrake pct vs Time', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('acc_vs_airbrake.png', dpi=300, bbox_inches='tight')
    print(f"Graph saved as 'acceleration.png'")
    print(f"Total data points: {len(time)}")
    print(f"Time range: {time[0]:.2f} to {time[-1]:.2f} seconds")

def plot_pressure():
    plt.figure()
    plt.plot(time, pressure)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pressure (hPa)')
    plt.title('Pressure vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pressure.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    # min_val = 5.875e5
    # max_val = 6.067e5

    min_val = 0
    max_val = 8.067e5
    min_index = None
    max_index = None
    for index, val in enumerate(time_data):
        if val > min_val and min_index is None:
            min_index = index
        if val > max_val and max_index is None:
            max_index = index

    print(min_index, max_index)
    time_data = time_data[min_index:max_index]
    xg_data = xg_data[min_index:max_index]
    yg_data = yg_data[min_index:max_index]
    zg_data = zg_data[min_index:max_index]
    yg_high_data = yg_high_data[min_index:max_index]
    airbrake_pct_data = airbrake_pct_data[min_index:max_index]
    plot_acc_vs_ab_pct(time_data, yg_data, yg_high_data, airbrake_pct_data)
    plot_acc(time_data, xg_data, yg_data, zg_data, airbrake_pct_data)
    # plot_pressure()