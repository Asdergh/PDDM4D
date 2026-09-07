import argparse

import numpy as np

import viser4d

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port", type=int, default=8080, help="Port to bind the server to."
)
args = parser.parse_args()

server = viser4d.Viser4dServer(num_steps=100, fps=10, port=args.port)

server.scene.add_frame("/origin", axes_length=0.25)
server.scene.add_grid("/ground", width=10.0, height=10.0)

point_cloud = None
for i in range(100):
    with server.at(i) as timeline:
        points = np.random.uniform(-1.0, 1.0, size=(200, 3))
        if point_cloud is None:
            point_cloud = timeline.scene.add_gaussian_splats(
                "/points",
                centers=points,
                covariances=np.random.normal(0, 0.3, (200, 3, 3)),
                rgbs=np.random.randint(0, 255, (200, 3)),
                opacities=np.random.normal(0, 1, (200, 1)),
            )
        else:
            point_cloud.centers += points
            point_cloud.covariances += np.random.normal(0, 0.03, (200, 3, 3))
            point_cloud.rgbs = np.random.randint(0, 255, (200, 3))
            point_cloud.opacities = np.random.normal(0, 1, (200, 1))

# Open the viewer and use the Playback controls in the GUI.
server.sleep_forever()