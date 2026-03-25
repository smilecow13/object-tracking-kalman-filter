import numpy as np

#generate the perfect 2D coordinates of a flying object
def generate_true_trajectory(v0=20.0, angle_deg=45.0, num_points=50):
    g = 9.81  
    angle_rad = np.radians(angle_deg)
    
    v0_x = v0 * np.cos(angle_rad)
    v0_y = v0 * np.sin(angle_rad)
    
    t_flight = 2 * v0_y / g
    
    t = np.linspace(0, t_flight, num=num_points)
    
    x = v0_x * t
    y = v0_y * t - 0.5 * g * t**2
    
    return t, x, y

#add noise to the true coordinates to create fake sensor readings
def add_measurement_noise(x, y, noise_std_dev=1.5):
   
    # np.random.normal(mean, standard_deviation, size)
    noise_x = np.random.normal(0, noise_std_dev, size=x.shape)
    noise_y = np.random.normal(0, noise_std_dev, size=y.shape)
    
    z_x = x + noise_x
    z_y = y + noise_y
    
    return z_x, z_y

#simulate losing track of the object .
def simulate_occlusion(z_x, z_y, start_index, end_index):
   
    z_x_occluded = np.copy(z_x)
    z_y_occluded = np.copy(z_y)
    
    z_x_occluded[start_index:end_index] = np.nan
    z_y_occluded[start_index:end_index] = np.nan
    
    return z_x_occluded, z_y_occluded

if __name__ == "__main__":
    t, true_x, true_y = generate_true_trajectory()
    measured_x, measured_y = add_measurement_noise(true_x, true_y, noise_std_dev=2.0)
    
    print(f"True X at step 10: {true_x[10]:.2f}m")
    print(f"Noisy X at step 10: {measured_x[10]:.2f}m")
