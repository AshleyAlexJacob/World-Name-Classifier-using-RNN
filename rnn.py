import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import  N_LETTERS
from utils import (
    load_data,
    line_to_tensor,
    random_training_example,
)
import os
from datetime import datetime
import dagshub
import mlflow

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden_layer(self):
        return torch.zeros(1, self.hidden_size)


def train(rnn, line_tensor, category_tensor, criterion, optimizer):
    hidden = rnn.init_hidden_layer()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

def save_model(model, name):
    os.makedirs(f"artifacts/models", exist_ok=True)
    torch.save(model.state_dict(), f"artifacts/models/{name}.pt")


def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

def predict(rnn, input_line):
    print(f"\n{input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        hidden = rnn.init_hidden_layer()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)

if __name__ == "__main__":
    category_lines, all_categories = load_data()
    n_categories = len(all_categories)
    n_hidden = 128
    dagshub.init("World-Name-Classifier-using-RNN", "alexjacob260", mlflow=True)
    
    rnn = RNN(N_LETTERS, n_hidden, n_categories)

    criterion = nn.NLLLoss()
    learning_rate = 0.005
    optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

    # training loop
    current_loss = 0
    all_losses = []
    # plot_steps, print_steps = 1000, 5000
    # n_iters = 100000

    plot_steps, print_steps = 1, 5
    n_iters = 10
    last_loss = 0
    mlflow.set_experiment('RNN_CLASSIFIER')
        
    with mlflow.start_run() as run:
        for i in range(n_iters):
            category, line, category_tensor, line_tensor = random_training_example(
                category_lines,
                all_categories,
            )
            output, loss = train(rnn, line_tensor, category_tensor,
                                criterion, optimizer)
            current_loss = loss

            if i==(n_iters-1): last_loss = current_loss


            if (i+1) % plot_steps==0:
                all_losses.append(current_loss/plot_steps)
                current_loss = 0
            
            if (i+1) % print_steps==0:
                guess = category_from_output(output)
                correct = f"Correct {guess}" if guess == category else f"Wrong {category}"
                print("f{i} {i/n_iters*100} {loss:.4f} {line}/{guess} {corect}")

            mlflow.log_metric("iterations",n_iters, step=i)
            mlflow.log_metric("loss", loss, step=i)
        
            
        # plt.figure()
        # plt.plot(all_losses)
        # plt.show()

        current_utc_date = datetime.utcnow()
        # save_model(rnn, f"classifier_{current_utc_date}")
        
        
        mlflow.pytorch.log_model(rnn, "classifier_{current_utc_date}")
        state_dict = rnn.state_dict()
        mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
        mlflow.log_param("lr", learning_rate)
        mlflow.log_param("hidden size", n_hidden)
            
            
    
    print(f"run_id: {run.info.run_id}")
    run_info = mlflow.get_run(run_id=run.info.run_id)
    print(f"params: {run_info.data.params}")
    print(f"metrics: {run_info.data.metrics}")


    # while True:
    #     sentence = input("Input:\t")
    #     if sentence=="exit":
    #         break

    #     predict(rnn, sentence)
        



# # one step
# input_tensor = letter_to_tensor('A')
# hidden_tensor = rnn.init_hidden_layer()
# output, hidden = rnn(input_tensor, hidden_tensor)
# print(output.size())
# print(hidden.size())

# # whole sequence/name
# input_tensor = line_to_tensor('Albert')
# hidden_tensor = rnn.init_hidden_layer()

# output, hidden = rnn(input_tensor[0], hidden_tensor)
# print(output.size())
# print(hidden.size())

# print(category_from_output(output))
