import pandas as pd
import torch


def read_data() -> pd.DataFrame:
    df = pd.read_csv("Students Social Media Addiction.csv")
    df = df.drop(["Student_ID", "Addicted_Score"], axis=1)

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Academic_Level"] = df["Academic_Level"].map({"High School":0, "Undergraduate":1, "Graduate":2})
    df["Relationship_Status"] = df["Relationship_Status"].map({"Single":0, "In Relationship":1, "Complicated":2})
    df["Affects_Academic_Performance"] = df["Affects_Academic_Performance"].map({"No":0, "Yes":1})
    df["Most_Used_Platform"] = df["Most_Used_Platform"].astype("category").cat.codes
    df["Country"] = df["Country"].astype("category").cat.codes

    numerical_cols = [
        "Age",
        "Avg_Daily_Usage_Hours",
        "Mental_Health_Score",
        "Conflicts_Over_Social_Media"
    ]
    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std

    return df

def create_dataset(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor(df["Sleep_Hours_Per_Night"].values, dtype=torch.float32).reshape(-1,1)
    input = torch.tensor(df.drop("Sleep_Hours_Per_Night", axis=1).values, dtype=torch.float32)
    return input, target

class FourLayerNN(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.l1(x))
        h2 = torch.tanh(self.l2(h1))
        o = torch.nn.functional.softplus(self.l3(h2))
        return o

def train_model(model: FourLayerNN, input: torch.Tensor, target: torch.Tensor):
    dataset = torch.utils.data.TensorDataset(input, target)
    loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(10000):
        for x, y_hat in loader:
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, y_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0:
            with torch.inference_mode():
                y = model(input)
                loss = torch.nn.functional.mse_loss(y, target)
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    df = read_data()
    input, target = create_dataset(df)

    model = FourLayerNN(input.shape[1],30)
    train_model(model, input, target)

    torch.save(model.state_dict(), "sleep_hours_model.pth")

    test_data = torch.tensor(
        [
            [
                21,  # Age
                1,   # Gender
                1,   # Academic Level
                0,   # Country
                8.0, # Avg_Daily_Usage_Hours
                0,   # Most_Used_Platform
                1,   # Affects_Academic_Performance
                3,   # Mental_Health_Score
                1,   # Relationship_Status
                2    # Conflicts_Over_Social_Media
            ]
        ],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        prediction = model(test_data)
        print(f"予測 Sleep_Hours_Per_Night: {prediction.item():.4f}")
