from rnn import RNN
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import mlflow
from utils import N_LETTERS
from utils import (
    random_training_example,
)
from datetime import datetime, timezone
import os
from dotenv import load_dotenv

class Trainer:
    def __init__(self, params, category_lines, all_categories):
        model = RNN(N_LETTERS, params.n_hidden, params.n_categories)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using Device", self.device)
        self.rnn = model.to(self.device)
        # if torch.cuda.is_available():
        #     print('Using Cuda')
        #     self.rnn = self.rnn.to('cuda')
        load_dotenv()
        mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
        self.all_categories = all_categories
        self.n_hidden = params.n_hidden
        self.category_lines = category_lines
        self.criterion = nn.NLLLoss()
        self.learning_rate = params.learning_rate
        self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.learning_rate)
    
    def utc_now(self):
        return datetime.utcnow().replace(tzinfo=timezone.utc)

    def category_from_output(self, output):
        category_idx = torch.argmax(output).item()
        return self.all_categories[category_idx]

    def __train(self, line_tensor, category_tensor):
        hidden = self.rnn.init_hidden_layer()
        hidden.to(self.device)
        line_tensor.to(self.device)
        category_tensor.to(self.device)
        # if torch.cuda.is_available():
        #     line_tensor = line_tensor.to('cuda')
        #     category_tensor = category_tensor.to('cuda')
        #     hidden = hidden.to('cuda')
            
        for i in range(line_tensor.size()[0]):
            output, hidden = self.rnn(line_tensor[i].to(self.device), hidden.to(self.device))

        loss = self.criterion(output, category_tensor)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return output, loss.item()

    def fit(self, config, params, visualize=False):
        current_loss = 0
        all_losses = []
        mlflow.set_experiment("RNN_CLASSIFIER_GPU")
        with mlflow.start_run() as run:
            for i in range(params.iterations):
                category, line, category_tensor, line_tensor = random_training_example(
                    self.category_lines,
                    self.all_categories,
                )
                output, loss = self.__train(line_tensor.to(self.device), category_tensor.to(self.device))
                current_loss = loss

                if (i + 1) % config.plot_steps == 0:
                    all_losses.append(current_loss / config.plot_steps)
                    current_loss = 0

                if (i + 1) % config.print_steps == 0:
                    guess = self.category_from_output(output)
                    correct = (
                        f"Correct {guess}" if guess == category else f"Wrong {category}"
                    )
                    print("f{i} {i/n_iters*100} {loss:.4f} {line}/{guess} {corect}")

                mlflow.log_metric("iterations", params.iterations, step=i)
                mlflow.log_metric("loss", loss, step=i)

            current_utc_date = self.utc_now()
            self.save_model(self.rnn, f"classifier_{current_utc_date}")
            mlflow.pytorch.log_model(self.rnn, "classifier_{current_utc_date}")
            state_dict = self.rnn.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")
            mlflow.log_param("lr", self.learning_rate)
            mlflow.log_param("hidden size", self.n_hidden)

        if visualize:
            plt.figure()
            plt.plot(all_losses)
            plt.show()

        return self.rnn

    def save_model(self, model, name):
        os.makedirs(f"artifacts/models", exist_ok=True)
        torch.save(model.state_dict(), f"artifacts/models/model.pt")
