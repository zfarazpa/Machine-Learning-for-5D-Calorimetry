import ROOT
import numpy as np
import math
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def determine_layer(cellCenterX, cellCenterY):
    radial_distance = math.sqrt(cellCenterX**2 + cellCenterY**2)

    if 2200 <= radial_distance < 2800:
        return "tile1"
    elif 2800 <= radial_distance < 3200:
        return "tile2"

    return None


class TimeCalibrationNN(nn.Module):
    def __init__(self, input_size):
        super(TimeCalibrationNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_nn(X, y):
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    mean_X = X.mean(axis=0)
    std_X = X.std(axis=0)
    std_X[std_X == 0] = 1.0

    X_scaled = (X - mean_X) / std_X

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=1234
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = TimeCalibrationNN(input_size=X.shape[1])

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 100

    for epoch in range(n_epochs):
        model.train()

        prediction = model(X_train)
        loss = loss_function(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_prediction = model(X_test)
                test_loss = loss_function(test_prediction, y_test)

            print(
                f"Epoch {epoch:3d} | "
                f"Train Loss = {loss.item():.6f} | "
                f"Test Loss = {test_loss.item():.6f}"
            )

    return model, mean_X, std_X


def main():
    input_file = ROOT.TFile.Open("../neutron.topo-cluster.001.new.eventNumberAdded.root")

    if not input_file or input_file.IsZombie():
        raise RuntimeError("Could not open input ROOT file.")

    tree = input_file.Get("CellTree")

    if not tree:
        raise RuntimeError("Could not find CellTree.")

    truthE_bins = [1, 5, 10, 15, 20, 30, 50, 100]
    eta_bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5]

    for iE in range(len(truthE_bins) - 1):
        truthE_min = truthE_bins[iE]
        truthE_max = truthE_bins[iE + 1]

        for iEta in range(len(eta_bins) - 1):
            eta_min = eta_bins[iEta]
            eta_max = eta_bins[iEta + 1]

            print("\n======================================")
            print(f"Running truthE {truthE_min}-{truthE_max}, eta {eta_min}-{eta_max}")
            print("======================================")

            X = []
            y = []
            selected_events = []

            for event in tree:
                cellE = event.cellE
                cellCenterX = event.cellCenterX
                cellCenterY = event.cellCenterY
                cellTime = event.cellTime
                truthE = event.truthE
                cellEta = event.cellEta

                if (
                    cellTime != 0
                    and cellE > 0.1
                    and truthE_min < truthE < truthE_max
                    and eta_min <= abs(cellEta) < eta_max
                ):
                    layer = determine_layer(cellCenterX, cellCenterY)

                    if layer is None:
                        continue

                    layer_id = 0 if layer == "tile1" else 1

                    features = [
                        cellE,
                        cellCenterX,
                        cellCenterY,
                        cellEta,
                        layer_id,
                    ]

                    X.append(features)
                    y.append(cellTime)
                    selected_events.append((features, cellTime, layer))

            if len(X) < 100:
                print(f"Not enough events: {len(X)}. Skipping this bin.")
                continue

            model, mean_X, std_X = train_nn(X, y)

            histograms = {
                "tile1": ROOT.TH1F(
                    f"tile1_truthE_{truthE_min}_{truthE_max}_eta_{eta_min}_{eta_max}",
                    f"NN Adjusted Time Tile 1, truthE {truthE_min}-{truthE_max}, eta {eta_min}-{eta_max};Adjusted Time (ns);Counts",
                    100,
                    -10,
                    10,
                ),
                "tile2": ROOT.TH1F(
                    f"tile2_truthE_{truthE_min}_{truthE_max}_eta_{eta_min}_{eta_max}",
                    f"NN Adjusted Time Tile 2, truthE {truthE_min}-{truthE_max}, eta {eta_min}-{eta_max};Adjusted Time (ns);Counts",
                    100,
                    -10,
                    10,
                ),
            }

            model.eval()

            with torch.no_grad():
                for features, cellTime, layer in selected_events:
                    features = np.array(features, dtype=np.float32)
                    features_scaled = (features - mean_X) / std_X

                    features_tensor = torch.tensor(
                        features_scaled.reshape(1, -1),
                        dtype=torch.float32
                    )

                    predicted_time = model(features_tensor).item()
                    adjusted_time = cellTime - predicted_time

                    histograms[layer].Fill(adjusted_time)

            for layer, histogram in histograms.items():
                if histogram.GetEntries() == 0:
                    continue

                canvas = ROOT.TCanvas(
                    f"canvas_{layer}_{truthE_min}_{truthE_max}_{eta_min}_{eta_max}",
                    f"NN Adjusted Time Fit {layer}",
                    800,
                    600,
                )

                histogram.Fit("gaus")
                histogram.Draw()

                output_name = (
                    f"nn_adjusted_time_fit_{layer}"
                    f"_truthE_{truthE_min}_{truthE_max}"
                    f"_eta_{eta_min}_{eta_max}.png"
                )

                canvas.SaveAs(output_name)

    input_file.Close()


if __name__ == "__main__":
    main()
