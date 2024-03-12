import torch
from src.utils import N_LETTERS, line_to_tensor, load_data, read_yaml
from src.rnn import RNN


class Predict:
    def __init__(self, model_path: str = None):
        self.params = read_yaml("params.yaml")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNN(N_LETTERS, self.params.n_hidden, self.params.n_categories)
        self.model.load_state_dict(
            torch.load("artifacts/models/state_dict.pth", map_location=self.device)
        )
        _, self.all_categories = load_data("data/names/")

    def category_from_output(self, output):
        category_idx = torch.argmax(output).item()
        return self.all_categories[category_idx]

    def predict(self, input_line):
        print(f"\n{input_line}")
        with torch.no_grad():
            line_tensor = line_to_tensor(input_line)
            hidden = self.model.init_hidden_layer()
            hidden.to(self.device)
            line_tensor.to(self.device)
            for i in range(line_tensor.size()[0]):
                output, hidden = self.model(line_tensor[i], hidden)
            guess = self.category_from_output(output)
            print(f"Guess: {guess}")


if __name__ == "__main__":
    predictor = Predict()
    predictor.predict("Ramsden")
