import numpy as np
from matplotlib import pyplot as plt


def gen_P_track(x1=5, y1=25, r1=5, name=None):
    first_sector = {
        'x': np.zeros(y1 - r1),
        'y': np.linspace(0, y1 - r1, y1 - r1)
    }
    
    sector2points_qty = 50 * r1
    theta2sector = np.pi - np.linspace(2 * np.pi / sector2points_qty, 3 * np.pi / 2 - 2 * np.pi / sector2points_qty, sector2points_qty - 1)
    second_sector = {
        'x': r1 * np.cos(theta2sector) + r1,
        'y': r1 * np.sin(theta2sector) + y1 - r1
    }
    
    third_sector = {
            'x': np.ones(y1 - 2 * r1) * r1,
            'y': np.linspace(0, y1 - 2 * r1, y1 - 2 * r1)
        }
    
    # plt.scatter(x=first_sector['x'], y=first_sector['y'])
    # plt.scatter(x=second_sector['x'], y=second_sector['y'])
    # plt.scatter(x=third_sector['x'], y=third_sector['y'])
    # plt.axis('equal')
    # plt.savefig('./track_editor/P_track.png')
    
    x_merged, y_merged = (
            np.concatenate((first_sector['x'], second_sector['x'], third_sector['x'])),
            np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
            )
    
    points = np.stack((x_merged, y_merged), axis=1)
    print(points.shape)
    return points, x_merged, y_merged

def gen_double_P_track(x1=5, y1=25, r1=5, name=None):
    first_sector = {
        'x': np.zeros(y1 - r1),
        'y': np.linspace(0, y1 - r1, y1 - r1)
    }
    
    sector2points_qty = 50 * r1
    theta2sector = np.pi - np.linspace(2 * np.pi / sector2points_qty, 3 * np.pi / 2 - 2 * np.pi / sector2points_qty, sector2points_qty - 1)
    second_sector = {
        'x': r1 * np.cos(theta2sector) + r1,
        'y': r1 * np.sin(theta2sector) + y1 - r1
    }
    
    third_sector = {
            'x': np.ones(y1 - 2 * r1) * r1,
            'y': np.linspace(0, y1 - 2 * r1, y1 - 2 * r1)
        }
    
    # plt.scatter(x=first_sector['x'], y=first_sector['y'])
    # plt.scatter(x=second_sector['x'], y=second_sector['y'])
    # plt.scatter(x=third_sector['x'], y=third_sector['y'])
    # plt.axis('equal')
    # plt.savefig('./track_editor/P_track.png')
    
    x_merged, y_merged = (
            np.concatenate((first_sector['x'], second_sector['x'], third_sector['x'])),
            np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
            )
    x_merged = np.concatenate((x_merged, x_merged + r1))
    y_merged = np.concatenate((y_merged, -y_merged))
    
    points = np.stack((x_merged, y_merged), axis=1)
    
    return points, x_merged, y_merged    


def gen_semi_circle_track(x1=15, y1=0, r1=15, name=None):
    
    sector1points_qty = 50 * r1
    theta1sector = np.pi - np.linspace(0, np.pi, sector1points_qty)
    first_sector = {
        'x': r1 * np.cos(theta1sector) + r1,
        'y': r1 * np.sin(theta1sector) + y1
    }
    
    # plt.scatter(x=first_sector['x'], y=first_sector['y'])
    # plt.scatter(x=second_sector['x'], y=second_sector['y'])
    # plt.scatter(x=third_sector['x'], y=third_sector['y'])
    # plt.axis('equal')
    # plt.savefig('./track_editor/P_track.png')
    
    # x_merged, y_merged = (
    #         np.concatenate((first_sector['x'], second_sector['x'], third_sector['x'])),
    #         np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
    #         )
    
    points = np.stack((first_sector['x'], first_sector['y']), axis=1)
    print(points.shape)
    return points, first_sector['x'], first_sector['y'] 
    
    
def gen_circle_track(x1=15, y1=0, r1=15, name=None):
    
    sector1points_qty = 50 * r1
    theta1sector = np.pi - np.linspace(0, 2 * np.pi - 2 * np.pi / sector1points_qty, sector1points_qty)
    first_sector = {
        'x': r1 * np.cos(theta1sector) + r1,
        'y': r1 * np.sin(theta1sector) + y1
    }
    
    # plt.scatter(x=first_sector['x'], y=first_sector['y'])
    # plt.scatter(x=second_sector['x'], y=second_sector['y'])
    # plt.scatter(x=third_sector['x'], y=third_sector['y'])
    # plt.axis('equal')
    # plt.savefig('./track_editor/P_track.png')
    
    # x_merged, y_merged = (
    #         np.concatenate((first_sector['x'], second_sector['x'], third_sector['x'])),
    #         np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
    #         )
    
    points = np.stack((first_sector['x'], first_sector['y']), axis=1)
    print(points.shape)
    return points, first_sector['x'], first_sector['y'] 
    
if __name__ == '__main__':
    gen_P_track()