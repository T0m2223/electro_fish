import pickle
import math
import sys
from operator import sub
import json

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import pandas as pd

import h5py

class Raycaster():
    """
    Raycast from fish to tank walls.
    """

    def __init__(self, corners, num_rays=6):
        """
        Input: corners: 4x2 list containing corner coords
        """
        if num_rays < 2:
            raise RuntimeError('Cannot have less than two rays')

        self.walls = [(corners[i], list(map(sub, corners[i-1], corners[i])))
                      for i in range(len(corners))]
        self.num_rays = num_rays
        self.step_angle = math.pi / (num_rays - 1)


    def crazy_formula(self, a, b, u, v):
        """
        Calculates distance to line defined by a and b from starting point point u with direction v.

        Input: Tensor[2]: a, u - absolute positions of WallCorner and FishHead respectively;
               Tensor[2]: b, v - direction vectors of wall and fish ray respectively
        Output: A distance started meassuring from point u to line ab, with direction v
        """
        return (((u[1] - a[1]) * b[0]) - ((u[0] - a[0]) * b[1])) / ((v[0] * b[1]) - (v[1] * b[0]))

    def __call__(self, head):
        """
        Calculates distance to walls for n provided angles.

        Input: Tensor[2 x 2] denoting head position and direction, respectively
        Output: Tensor[n] containing distances
        """
        head = head.clone()
        # Normalize fish facing vector
        head[1] /= math.sqrt(sum(x ** 2 for x in head[1]))

        smallest_dist = torch.full((self.num_rays,), float('inf'))

        head[1] = rotate_tensor((math.pi / 2) + self.step_angle, head[1])
        for i in range(self.num_rays):
            head[1] = rotate_tensor(-self.step_angle, head[1])
            for wall_a, wall_b in self.walls:
                dist = self.crazy_formula(wall_a, wall_b, head[0], head[1])
                if dist > 0:
                    smallest_dist[i] = min(smallest_dist[i], dist)

        return smallest_dist




def rotation_angle(vector):
    """
    Given an XY vector, calculate its angle between X-axis.

    Input: 2D-Tensor
    Output: Scalar angle in radians
    """
    # Calculate the angle using arctangent
    return np.arctan2(vector[1].item(), vector[0].item())

def rotate_tensor(angle, tensor):
    """
    Rotates given Tensor along XY-axis.
    The resulting tensor will have its own memory.

    Input: angle in radians (-pi/2 to pi/2), Tensor[.., 2]
    Output: Tensor, with values rotated by angle
    """
    # Convert the angle to a tensor
    angle_tensor = torch.tensor([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]], dtype=tensor.dtype)
    # Rotate the tensor
    return torch.matmul(tensor.clone(), angle_tensor)


def egocentric_transform(sample):
    """
    Transform the given datapoints into a pytorch vector containing egocentric views
    with respect to the other fish for both fish.

    Input: Tensor[2,6,2] (Fish x Track point x XY)
    Output: Tuple(Tensor[2,6,2], Tensor[2,6,2]) (Egocentric views, on for each fish)
    """
    # Center positions and calculate angles
    centered2 = (sample - sample[1, 5]).clone()
    centered1 = sample - sample[0, 5]
    angle1, angle2 = rotation_angle(centered1[0][0]), rotation_angle(centered2[1][0])

    # Put positions of second fish in front
    centered2[0], centered2[1] = centered2[1].clone(), centered2[0].clone()

    # Rotate vectors
    return rotate_tensor(-angle1, centered1), rotate_tensor(-angle2, centered2)


def generate_label(frame1, frame2):
    """
    Generates labels for tracking points, relative center movement and rotation
    based upon two provided, consecutive frames (datapoints).

    Input: two Tensor[2,6,2] (Fish x Track point x XY)
    Output: dict( d_center:   Tensor[2,2] (Fish x XY) (center diff relative to frame 1 rotation),
                  d_rotation: Tensor[2] (radians diff for each fish),
                  d_points:   Tensor[2,6,2] (Fish x Points x XY) (point diffs disregarding rotation
                                                                  and position) )
    """

    center_diff = frame2[:, 5, :] - frame1[:, 5, :]

    center_diff_rel = torch.zeros(2, 2)
    body_diff_rel = torch.zeros(2, 6, 2)
    rotation_rel = torch.zeros(2)

    for i in range(2):
        # Centered by current fish center
        centered1 = frame1[i] - frame1[i, 5]
        centered2 = frame2[i] - frame2[i, 5]

        # Angles of both frames
        rot1, rot2 = rotation_angle(centered1[0]), rotation_angle(centered2[0])

        center_diff_rel[i] = rotate_tensor(-rot1, center_diff[i])
        body_diff_rel[i] = rotate_tensor(-rot2, centered2) - rotate_tensor(-rot1, centered1)
        rotation_rel[i] = rot2 - rot1

    return {'d_center'  : center_diff_rel,
            'd_rotation': rotation_rel,
            'd_points'  : body_diff_rel}


def decode_next_position(body_pos_abs, center_diff_rel, rotation_rel, body_diff_rel):
    """
    Calculates new body positions, given initial positions and network output.

    Input: body_pos_abs:    Tensor[2,6,2] (Fish x Track point x XY),
           center_diff_rel: Tensor[2,2] (Fish x XY) (center diff relative to initial rotation),
           rotation_rel:    Tensor[2] (radians diff for each fish),
           body_diff_rel:   Tensor[2,6,2] (Fish x Points x XY) (point diffs)
    Output: Tensor[2,6,2] (Fish x Track point x XY)
    """
    result = torch.zeros(2, 6, 2)
    for i in range(2):
        centered_old = body_pos_abs[i] - body_pos_abs[i][5]
        angle_old = rotation_angle(centered_old[0])
        # Apply body diff and rotation
        result[i] = rotate_tensor(angle_old + rotation_rel[i], centered_old + body_diff_rel[i])
        # Translate back into world coords
        result[i] += body_pos_abs[i][5] + rotate_tensor(angle_old, center_diff_rel[i])

    return result


def input_from_position(position, raycaster):
    inp1, inp2 = egocentric_transform(position)
    ray1 = raycaster(torch.stack((position[0][0], position[0][0] - position[0][5])))
    ray2 = raycaster(torch.stack((position[1][0], position[1][0] - position[1][5])))
    return torch.cat([inp1.view(-1), ray1]), torch.cat([inp2.view(-1), ray2])

def squish_label(d_center, d_rotation, d_points):
    """
    Concats all the data into a 1D Tensor.
    """
    return torch.cat([d_center.view(-1), d_rotation.view(-1), d_points.view(-1)])

def unsquish_label(mixed):
    """
    Reverses the concat done by squish_label.
    """
    d_center = mixed[:4].view(2, 2)
    d_rotation = mixed[4:6].view(2)
    d_points = mixed[6:30].view(2, 6, 2)

    return {'d_center'  : d_center,
            'd_rotation': d_rotation,
            'd_points'  : d_points}

class FishDataset(Dataset):
    """
    Dataset for fish data, applies egocentric transforms and generates labels.
    Each item is a tuple for each fish (2).
    These tuples consist of input data Tensor[2,6,2] (Fish x Points x XY)
    and label Tensor[30] lab: (
        d_center:   lab[:4] (Fish(2) x XY(2)),
        d_rotation: lab[4:6] (2),
        d_points:   lab[6:30] (Fish(2) x Points(6) x XY(2)) )
    """

    def __init__(self, corners, tracks=None, file_path=None, preprocess=True):
        self.tracks = tracks
        self.raycaster = Raycaster(corners)
        self.file_path = file_path
        self.preprocess = preprocess if not file_path is None else True
        if self.preprocess and tracks is None:
            raise RuntimeError('Cannot preprocess without track data')

        self.data = self._load_data()

    def __len__(self):
        return (len(self.tracks) - 1) if self.preprocess else (len(self.data) - 1)


    def _preprocess(self):
        """
        Preprocess data, only use if preprocess enable.
        """
        data = []

        for i in range(len(self)):
            frame1, frame2 = self.tracks[i], self.tracks[i+ 1]

            inp1, inp2 = input_from_position(frame1, self.raycaster)
            label = generate_label(frame1, frame2)
            lab1 = squish_label(label['d_center'][0], label['d_rotation'][0], label['d_points'][0])
            lab2 = squish_label(label['d_center'][1], label['d_rotation'][1], label['d_points'][1])

            data.append((inp1, lab1, inp2, lab2))

        return data

    def _load_data(self):
        if self.preprocess:
            data = self._preprocess()
            with open(self.file_path, 'wb') as file:
                pickle.dump(data, file)
        else:
            with open(self.file_path, 'rb') as file:
                data = pickle.load(file)
        return data


    def __getitem__(self, idx):
        return self.data[idx]



class SequentialDataset(Dataset):
    """
    Dataset to possibly wrap another one to get sequences of data instead of single data points.
    """

    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):
        inputs1, inputs2 = [], []
        labels1, labels2 = [], []
        for i in range(idx, idx + self.sequence_length):
            inp1, lab1, inp2, lab2 = self.data[i]
            inputs1.append(inp1)
            inputs2.append(inp2)
            labels1.append(lab1)
            labels2.append(lab2)

        return [torch.stack(inputs1), torch.stack(labels1),
                torch.stack(inputs2), torch.stack(labels2)]


def visualize(data_creator, tank_corners, file_path):
    """
    Creates a video based on provided fish positions.

    Input: random access frame provider, e.g. Tensor[frames,2,6,2], tank corners, file save path
    """
    fig, axis = plt.subplots()

    axis.set_xlim(min(c[0] for c in tank_corners), max(c[0] for c in tank_corners))
    axis.set_ylim(min(c[1] for c in tank_corners), max(c[1] for c in tank_corners))

    scatter_plots = [axis.scatter([], [], label=f"Fish {i+1}") for i in range(2)]

    lines = [axis.plot([], [], color=plot.get_facecolors()[0].tolist(), linestyle='-', linewidth=2)
             [0] for plot in scatter_plots]

    def animate(frame):
        for i, plot in enumerate(scatter_plots):
            positions = data_creator[frame][i].numpy()
            plot.set_offsets(positions)
            indices_to_connect = [0, 5, 1, 5, 2, 5, 3, 4]
            x_connect = [positions[point][0] for point in indices_to_connect]
            y_connect = [positions[point][1] for point in indices_to_connect]
            lines[i].set_data(x_connect, y_connect)

        if frame % 100 == 99:
            print(f"Frame {frame}")
        return *scatter_plots, *lines

    ani = animation.FuncAnimation(fig, animate, frames=len(data_creator), interval=200, blit=True)
    plt.legend()
    ani.save(file_path, writer='ffmpeg', fps=20)



class TransformerModel(nn.Module):
    def __init__(self, input_size=60, hidden_size=100, num_layers=15, num_heads=4, output_size=30):
        super(TransformerModel, self).__init__()

        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=0.1
        )

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Assuming x has shape (seq_len, batch_size, input_size)
        x = self.transformer(x, x)
        x = self.fc(x)
        return x


def train(model, tank_corners, sequence_length=128, batch_size=128, epochs=2):
    """
    Training function.
    """
    #dataset = FishDataset(tank_corners, tracks=tracks_matrix, file_path='data/Mormyrus_Pair_01/preprocessed.bin')
    dataset = FishDataset(tank_corners, file_path='data/Mormyrus_Pair_01/preprocessed.bin', preprocess=False)
    dataset = SequentialDataset(dataset, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())
    
    for i in epochs:
        for inp1, lab1, inp2, lab2 in dataloader:
            model_in = torch.cat(inp1, inp2)
            label = torch.cat(lab1, lab2)


            optimizer.zero_grad()  

            model_out = model(model_in)
            loss = torch.MSELoss(model_out, label)
            loss.backward()
            optimizer.step()

        print(f"loss: {loss}")


def simulate(model, fish_pos, tank_corners, num_steps=1000):
    raycaster = Raycaster(tank_corners)

    for i in range(num_steps):
        model_in = input_from_position(fish_pos, raycaster)
        model_out = model(model_in)
        out1, out2 = unsquish_label(model_out[0]), unsquish_label(model_out[1])
        print(out1.shape)
        new_pos = decode_next_position(last_pos)


TRACK_FILE = 'data/Mormyrus_Pair_01/poses/20230316_Mormyrus_Pair_01.000_20230316_Mormyrus_Pair_01.analysis.h5'
TANK_ANNOTATIONS_FILE = 'data/Mormyrus_Pair_01/metadata/tank_annotations.json'

print('Reading tracks file...', end='')
sys.stdout.flush()
with h5py.File(TRACK_FILE, 'r') as f:
    occupancy_matrix = f['track_occupancy'][:]
    tracks_matrix = torch.tensor(f['tracks'][:]).permute(3, 0, 2, 1)
print('OK')

print('Reading tank annotation file...', end='')
sys.stdout.flush()
with open(TANK_ANNOTATIONS_FILE, 'r') as json_file:
    tank_corner_data = json.load(json_file)
print('OK')

print('Getting tank dimensions...', end='')
sys.stdout.flush()
TANK_CORNERS = []
for shape in tank_corner_data.get("shapes", []):
    if shape.get("label") == "TankCorners":
        TANK_CORNERS = shape.get("points", [])
        break
print('OK')

fishModel = TransformerModel()
train(fishModel, TANK_CORNERS, sequence_length=128, batch_size=256)
#visualize(tracks_matrix[:400], TANK_CORNERS, 'fish_animation.mp4')
