import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper, feedback_lin_ctrl, SinusoidalReference
import statsmodels.api as sm
from sklearn.metrics import r2_score


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig_DR.json"  # Configuration file for the robot
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)  # Initialize simulation interface

    # Get active joint names from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names = np.expand_dims(np.array(ext_names), axis=0)  # Adjust the shape for compatibility

    source_names = ["pybullet"]  # Define the source for dynamic modeling

    # Create a dynamic model of the robot
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()

    # Print initial joint angles
    print(f"Initial joint angles: {sim.GetInitMotorAngles()}")

    # Sinusoidal reference
    # Specify different amplitude values for each joint
    amplitudes = [np.pi / 4, np.pi / 6, np.pi / 4, np.pi / 4, np.pi / 4, np.pi / 4,
                  np.pi / 4]  # Example amplitudes for joints
    # Specify different frequency values for each joint
    frequencies = [0.4, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4]  # Example frequencies for joints

    # Convert lists to NumPy arrays for easier manipulation in computations
    amplitude = np.array(amplitudes)
    frequency = np.array(frequencies)
    ref = SinusoidalReference(amplitude, frequency, sim.GetInitMotorAngles())  # Initialize the reference

    # Simulation parameters
    time_step = sim.GetTimeStep()
    current_time = 0
    max_time = 10  # seconds

    # Command and control loop
    cmd = MotorCommands()  # Initialize command structure for motors
    # PD controller gains
    kp = 1000
    kd = 100

    # Initialize data storage
    tau_mes_all = []
    regressor_all = []

    # Data collection loop
    while current_time < max_time:
        # Measure current state
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_mes = sim.ComputeMotorAccelerationTMinusOne(0)

        # Compute sinusoidal reference trajectory
        q_d, qd_d = ref.get_values(current_time)  # Desired position and velocity

        # Control command
        cmd.tau_cmd = feedback_lin_ctrl(dyn_model, q_mes, qd_mes, q_d, qd_d, kp, kd)
        sim.Step(cmd, "torque")

        # Get measured torque
        tau_mes = sim.GetMotorTorques(0)

        if dyn_model.visualizer:
            for index in range(len(sim.bot)):  # Conditionally display the robot model
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)  # Update the display of the robot model

        # Exit logic with 'q' key
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Compute regressor and store it
        cur_regressor = dyn_model.ComputeDynamicRegressor(q_mes, qd_mes, qdd_mes) # 7, 70
        if current_time > 0.5:
            regressor_all.append(cur_regressor) #
            tau_mes_all.append(tau_mes) #

        current_time += time_step
        # Optional: print current time
        print(f"Current time in seconds: {current_time:.2f}")

    # After data collection, stack all the regressor and all the torquen and compute the parameters 'a'  using pseudoinverse
    sample_num = len(tau_mes_all)
    print("sample_num", sample_num)
    regressor_all_stacked = np.vstack(regressor_all)
    print("regressor_all_stacked: ", regressor_all_stacked.shape)
    tau_mes_all_stacked = np.hstack(tau_mes_all)
    print("tau_mes_all_stacked: ", tau_mes_all_stacked.shape)

    a = np.linalg.pinv(regressor_all_stacked).dot(tau_mes_all_stacked)
    print(a)

    # TODO compute the metrics for the linear model
    # regressor_all_stacked = sm.add_constant(regressor_all_stacked)  # 添加常数项
    model = sm.OLS(tau_mes_all_stacked, regressor_all_stacked).fit()  # 使用普通最小二乘法进行拟合

    # 计算 adjusted R-squared
    r_squared_adj = model.rsquared_adj
    print(f"Adjusted R-squared: {r_squared_adj}")

    # 计算 F-statistics
    f_statistic = model.fvalue
    print(f"F-statistic: {f_statistic}")

    # 打印置信区间
    confidence_intervals = model.conf_int()
    print(f"Confidence intervals: \n{confidence_intervals}")


    # MSE
    MSE = model.mse_model
    print(f"MSE: {MSE}")

    # 打印完整的回归结果汇总
    print(model.summary())

    # TODO plot the  torque prediction error for each joint (optional)
    # 使用回归矩阵 Y 和系数向量 a 计算预测的扭矩
    u_predicted = regressor_all_stacked.dot(a)

    # 重新调整形状，使其对应7个关节的扭矩数据，每个关节有 10001 组样本
    u_original_joint = tau_mes_all_stacked.reshape(sample_num, 7)
    u_predicted_joint = u_predicted.reshape(sample_num, 7)

    # 创建一个图形，大小为 (10, 14)，包含 7 个子图，每个子图代表一个关节
    fig, axs = plt.subplots(7, 1, figsize=(10, 14))

    # 为每个关节绘制原始扭矩和预测扭矩的曲线
    for joint in range(7):
        axs[joint].plot(u_original_joint[:, joint], label="Original Torque", color='b', linewidth=1)
        axs[joint].plot(u_predicted_joint[:, joint], label="Predicted Torque", color='r', linestyle='--', linewidth=1)
        axs[joint].set_title(f"Joint {joint + 1}")
        axs[joint].set_xlabel("Sample")
        axs[joint].set_ylabel("Torque")
        axs[joint].legend()
        axs[joint].grid(True)

    # 自动调整子图布局
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
