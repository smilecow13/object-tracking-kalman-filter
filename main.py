import matplotlib.pyplot as plt
import data_generate as dg

def main():
    #generate the true trajectory
    t, true_x, true_y = dg.generate_true_trajectory(v0=20.0, angle_deg=45.0, num_points=50)
    #add noise to the true trajectory
    measured_x, measured_y = dg.add_measurement_noise(true_x, true_y, noise_std_dev=1.5)

    #visualize the data
    plt.figure(figsize=(10, 6))
    plt.plot(true_x, true_y, label='True Trajectory', color='green', linewidth=2)
    plt.scatter(measured_x, measured_y, label='Noisy Camera Data', color='red', marker='x', alpha=0.7)

    plt.title('Object Trajectory: Ground Truth vs Noisy Sensor Data')
    plt.xlabel('Distance X (m)')
    plt.ylabel('Height Y (m)')
    plt.axhline(0, color='black', linewidth=1) 
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    main()