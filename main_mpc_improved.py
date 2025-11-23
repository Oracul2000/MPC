"""
Test script for improved MPC controller.
Compares old MPC vs new improved MPC with cross track error constraints.
"""

from sim.py_model.model import *
from sim.track_editor.create import *

from controls.mpc.mpc import MPCController
from controls.mpc.mpc_improved import MPCControllerImproved

from matplotlib import pyplot as plt
import numpy as np
import time


def run_simulation(controller, car, path, dt=0.05, max_steps=8000, v_ref=10.0, name="Controller"):
    """Run simulation with given controller"""
    print(f"\n{'='*60}")
    print(f"Running simulation with {name}")
    print(f"{'='*60}")
    
    vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'vx': [], 'vy': [], 'cte': [], 'time': []}
    control_history = {'throttle': [], 'steering': [], 'time': []}
    
    step_times = []
    start_time = time.time()
    
    for step in range(max_steps):
        x, y, yaw = car.getX(), car.getY(), car.getyaw()
        vx, vy, w = car.getvx(), car.getvy(), car.getr()
        
        # Compute control
        step_start = time.time()
        
        if isinstance(controller, MPCControllerImproved):
            # New controller uses full state
            throttle, steering = controller.compute_control(x, y, yaw, vx, vy, w)
        else:
            # Old controller uses simplified state
            v = np.sqrt(vx**2 + vy**2)
            throttle, steering = controller.compute_control(x, y, yaw, v, v_ref)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Apply control
        brakes = 0.0
        if throttle < 0:
            brakes = -throttle
            throttle = 0.0
        
        u = ControlInfluence(throttle, steering, brakes)
        car.updateRK4(u)
        
        # Compute Cross Track Error
        if isinstance(controller, MPCControllerImproved):
            _, idx, cte = controller.find_closest_point(x, y)
        else:
            distances = np.sqrt((path[:, 0] - x)**2 + (path[:, 1] - y)**2)
            cte = np.min(distances)
            idx = np.argmin(distances)
        
        # Store trajectory
        vehicle_traj['x'].append(x)
        vehicle_traj['y'].append(y)
        vehicle_traj['yaw'].append(yaw)
        vehicle_traj['vx'].append(vx)
        vehicle_traj['vy'].append(vy)
        vehicle_traj['cte'].append(cte)
        vehicle_traj['time'].append(step * dt)
        
        control_history['throttle'].append(throttle + brakes * (-1))
        control_history['steering'].append(steering)
        control_history['time'].append(step * dt)
        
        # Progress update
        if step % 1000 == 0:
            print(f"Step {step}/{max_steps}: x={x:.1f}, y={y:.1f}, vx={vx:.1f} m/s, "
                  f"CTE={cte:.2f} m, avg_step_time={np.mean(step_times[-100:]):.4f}s")
        
        # Check if car went off track badly
        if cte > 10.0:
            print(f"⚠️  Vehicle went off track! CTE={cte:.2f} m at step {step}")
            break
    
    total_time = time.time() - start_time
    
    # Print statistics
    print(f"\n{name} Statistics:")
    print(f"  Total simulation time: {total_time:.2f} s")
    print(f"  Average step time: {np.mean(step_times):.4f} s")
    print(f"  Max step time: {np.max(step_times):.4f} s")
    print(f"  Average speed: {np.mean(vehicle_traj['vx']):.2f} m/s")
    print(f"  Max speed: {np.max(vehicle_traj['vx']):.2f} m/s")
    print(f"  Average CTE: {np.mean(vehicle_traj['cte']):.3f} m")
    print(f"  Max CTE: {np.max(vehicle_traj['cte']):.3f} m")
    print(f"  RMS CTE: {np.sqrt(np.mean(np.array(vehicle_traj['cte'])**2)):.3f} m")
    
    if isinstance(controller, MPCControllerImproved):
        stats = controller.get_statistics()
        if stats:
            print(f"  MPC success rate: {stats['success_rate']:.1f}%")
            print(f"  MPC avg solve time: {stats['avg_solve_time']:.4f} s")
    
    return vehicle_traj, control_history


def plot_comparison(path, center_line_x, center_line_y, 
                   traj_old, traj_new, 
                   control_old, control_new):
    """Plot comparison between old and new MPC"""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Trajectory comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(center_line_x, center_line_y, 'g--', linewidth=2, label='Reference Path', alpha=0.7)
    ax1.plot(traj_old['x'], traj_old['y'], 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax1.plot(traj_new['x'], traj_new['y'], 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory Comparison')
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Trajectory zoom (first section)
    ax2 = plt.subplot(3, 3, 2)
    n_zoom = min(1000, len(traj_old['x']))
    ax2.plot(center_line_x[:n_zoom], center_line_y[:n_zoom], 'g--', linewidth=2, label='Reference', alpha=0.7)
    ax2.plot(traj_old['x'][:n_zoom], traj_old['y'][:n_zoom], 'b-', linewidth=2, label='Old MPC')
    ax2.plot(traj_new['x'][:n_zoom], traj_new['y'][:n_zoom], 'r-', linewidth=2, label='Improved MPC')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Trajectory Zoom (First Section)')
    ax2.legend()
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. Speed comparison
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(traj_old['time'], traj_old['vx'], 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax3.plot(traj_new['time'], traj_new['vx'], 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax3.axhline(y=np.mean(traj_old['vx']), color='b', linestyle=':', alpha=0.5, label=f"Old Avg: {np.mean(traj_old['vx']):.2f}")
    ax3.axhline(y=np.mean(traj_new['vx']), color='r', linestyle=':', alpha=0.5, label=f"New Avg: {np.mean(traj_new['vx']):.2f}")
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Longitudinal Velocity Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross Track Error comparison
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(traj_old['time'], traj_old['cte'], 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax4.plot(traj_new['time'], traj_new['cte'], 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax4.axhline(y=2.0, color='orange', linestyle='--', label='Max CTE Constraint (2.0m)')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Cross Track Error (m)')
    ax4.set_title('Cross Track Error Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Lateral velocity comparison
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(traj_old['time'], traj_old['vy'], 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax5.plot(traj_new['time'], traj_new['vy'], 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Lateral Velocity (m/s)')
    ax5.set_title('Lateral Velocity (Stability Indicator)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Throttle comparison
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(control_old['time'], control_old['throttle'], 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax6.plot(control_new['time'], control_new['throttle'], 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Throttle')
    ax6.set_title('Throttle Control Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Steering comparison
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(control_old['time'], np.rad2deg(control_old['steering']), 'b-', linewidth=1.5, label='Old MPC', alpha=0.7)
    ax7.plot(control_new['time'], np.rad2deg(control_new['steering']), 'r-', linewidth=1.5, label='Improved MPC', alpha=0.7)
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Steering Angle (deg)')
    ax7.set_title('Steering Control Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Control rate (smoothness) - throttle
    ax8 = plt.subplot(3, 3, 8)
    throttle_rate_old = np.diff(control_old['throttle']) / 0.05
    throttle_rate_new = np.diff(control_new['throttle']) / 0.05
    ax8.plot(control_old['time'][1:], throttle_rate_old, 'b-', linewidth=1, label='Old MPC', alpha=0.7)
    ax8.plot(control_new['time'][1:], throttle_rate_new, 'r-', linewidth=1, label='Improved MPC', alpha=0.7)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Throttle Rate (1/s)')
    ax8.set_title('Throttle Rate (Smoothness)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Control rate (smoothness) - steering
    ax9 = plt.subplot(3, 3, 9)
    steering_rate_old = np.rad2deg(np.diff(control_old['steering'])) / 0.05
    steering_rate_new = np.rad2deg(np.diff(control_new['steering'])) / 0.05
    ax9.plot(control_old['time'][1:], steering_rate_old, 'b-', linewidth=1, label='Old MPC', alpha=0.7)
    ax9.plot(control_new['time'][1:], steering_rate_new, 'r-', linewidth=1, label='Improved MPC', alpha=0.7)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Steering Rate (deg/s)')
    ax9.set_title('Steering Rate (Smoothness)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mpc_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: mpc_comparison.png")
    
    # Performance summary
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
    MPC CONTROLLER COMPARISON
    {'='*60}
    
    AVERAGE SPEED:
    • Old MPC:       {np.mean(traj_old['vx']):.2f} m/s
    • Improved MPC:  {np.mean(traj_new['vx']):.2f} m/s
    • Improvement:   {(np.mean(traj_new['vx']) - np.mean(traj_old['vx'])):.2f} m/s ({((np.mean(traj_new['vx']) / np.mean(traj_old['vx'])) - 1) * 100:.1f}%)
    
    MAX SPEED:
    • Old MPC:       {np.max(traj_old['vx']):.2f} m/s
    • Improved MPC:  {np.max(traj_new['vx']):.2f} m/s
    
    CROSS TRACK ERROR (CTE):
    • Old MPC RMS:       {np.sqrt(np.mean(np.array(traj_old['cte'])**2)):.3f} m
    • Improved MPC RMS:  {np.sqrt(np.mean(np.array(traj_new['cte'])**2)):.3f} m
    • Old MPC Max:       {np.max(traj_old['cte']):.3f} m
    • Improved MPC Max:  {np.max(traj_new['cte']):.3f} m
    
    CONTROL SMOOTHNESS (RMS Rate):
    • Old Throttle Rate:      {np.sqrt(np.mean(throttle_rate_old**2)):.3f} 1/s
    • Improved Throttle Rate: {np.sqrt(np.mean(throttle_rate_new**2)):.3f} 1/s
    • Old Steering Rate:      {np.sqrt(np.mean(steering_rate_old**2)):.3f} deg/s
    • Improved Steering Rate: {np.sqrt(np.mean(steering_rate_new**2)):.3f} deg/s
    
    LATERAL STABILITY (RMS vy):
    • Old MPC:       {np.sqrt(np.mean(np.array(traj_old['vy'])**2)):.3f} m/s
    • Improved MPC:  {np.sqrt(np.mean(np.array(traj_new['vy'])**2)):.3f} m/s
    """
    
    ax.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=11, 
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('mpc_summary.png', dpi=150, bbox_inches='tight')
    print(f"✓ Summary saved to: mpc_summary.png")


def main():
    print("="*60)
    print("MPC CONTROLLER COMPARISON TEST")
    print("="*60)
    
    # Generate track
    print("\n1. Generating track...")
    path, center_line_x, center_line_y = gen_double_P_track()
    print(f"   Track has {len(path)} waypoints")
    
    # Test parameters
    dt = 0.05
    v_ref = 15.0  # Higher target speed
    max_steps = 4000  # Shorter for faster comparison
    horizon = 15
    
    # ========================================
    # Test 1: Old MPC
    # ========================================
    print("\n2. Testing OLD MPC...")
    car_old = Dynamic4WheelsModel()
    car_old.carState.body.yaw = np.pi / 2
    mpc_old = MPCController(path, wheelbase=1.5, horizon=horizon, dt=dt)
    
    traj_old, control_old = run_simulation(
        mpc_old, car_old, path, dt=dt, max_steps=max_steps, v_ref=v_ref, name="Old MPC"
    )
    
    # ========================================
    # Test 2: Improved MPC
    # ========================================
    print("\n3. Testing IMPROVED MPC...")
    car_new = Dynamic4WheelsModel()
    car_new.carState.body.yaw = np.pi / 2
    mpc_new = MPCControllerImproved(
        path, 
        wheelbase=1.5, 
        horizon=horizon, 
        dt=dt,
        max_cte=2.0,  # Cross track error constraint
        max_speed=30.0,  # Maximum speed
        verbose=False  # Set True for detailed output
    )
    
    traj_new, control_new = run_simulation(
        mpc_new, car_new, path, dt=dt, max_steps=max_steps, v_ref=v_ref, name="Improved MPC"
    )
    
    # ========================================
    # Plot comparison
    # ========================================
    print("\n4. Generating comparison plots...")
    plot_comparison(path, center_line_x, center_line_y, 
                   traj_old, traj_new,
                   control_old, control_new)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print("\nKey Improvements in New MPC:")
    print("  ✓ Dynamic vehicle model (vx, vy, w)")
    print("  ✓ Cross Track Error constraints")
    print("  ✓ Speed maximization objective")
    print("  ✓ Warm start optimization")
    print("  ✓ Control rate penalty (smoother)")
    print("  ✓ Better reference trajectory")
    print("  ✓ Optimization success checking")
    print("  ✓ Fallback control strategy")
    print("\nCheck generated images:")
    print("  • mpc_comparison.png - Detailed comparison plots")
    print("  • mpc_summary.png - Performance summary")


if __name__ == '__main__':
    main()
