import torch
from utils import line_to_tensor


class Predict:
    def category_from_output(self, output):
        category_idx = torch.argmax(output).item()
        return self.all_categories[category_idx]

    def predict(self, rnn, input_line):
        print(f"\n{input_line}")
        with torch.no_grad():
            line_tensor = line_to_tensor(input_line)
            hidden = rnn.init_hidden_layer()
            for i in range(line_tensor.size()[0]):
                output, hidden = rnn(line_tensor[i], hidden)

            guess = self.category_from_output(output)
            print(guess)
