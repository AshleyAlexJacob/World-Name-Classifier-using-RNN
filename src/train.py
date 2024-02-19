from trainer import Trainer
from utils import load_data, read_yaml
import dagshub
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv(".env")
    params = read_yaml("params.yaml")
    config = read_yaml("config.yaml")

    category_lines, all_categories = load_data(config.data)
    # dagshub.init(
    #     os.environ.get("dagshub_repo"), os.environ.get("dagshub_user"), mlflow=True
    # )

    trainer = Trainer(params, category_lines, all_categories)
    model = trainer.fit(config, params)
