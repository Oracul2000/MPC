import numpy as np


def find_lookahead_point(current_pos, centerLine, R):
    distances = np.linalg.norm(centerLine - current_pos, axis=1)
    idx = np.argmin(distances)
    for i in range(idx, len(centerLine)):
        if distances[i] >= R:
            return centerLine[i]
    return centerLine[-1]

def steering_controller(model, centerLine, R):
    current_pos = np.array([model.carState.body.X, model.carState.body.Y])
    lookahead = find_lookahead_point(current_pos, centerLine, R)
    vector_to_lookahead = lookahead - current_pos
    angle_to_lookahead = np.arctan2(vector_to_lookahead[1], vector_to_lookahead[0])
    steering_angle = angle_to_lookahead - model.carState.body.yaw
    steering_angle = np.clip(steering_angle, -np.pi / 2, np.pi / 2)  # Limit steering
    return steering_angle