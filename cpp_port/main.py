import car_model
import matplotlib.pyplot as plt

def run_simulation():
    """
    Запускает симуляцию модели автомобиля и строит траекторию.
    """
    # Создаем экземпляр модели
    model = car_model.Dynamic4WheelsModel()

    controls = car_model.ControlInfluence(throttle=0.5, steeringAngle=0.2, brakes=0.0)
    
    history_x = []
    history_y = []

    iterations = 5000 
    
    print("Starting simulation...")
    for i in range(iterations):
        model.update(controls)
        
        current_state = model.get_state()
        
        history_x.append(current_state.X)
        history_y.append(current_state.Y)

        if i % 500 == 0:
            print(f"Time: {model.t:.2f}s, State: {current_state}")
            
    print("Simulation finished.")
    
    # Визуализация результатов
    plt.figure(figsize=(10, 8))
    plt.plot(history_x, history_y, label='Car Trajectory')
    plt.title('Vehicle Simulation')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_simulation()