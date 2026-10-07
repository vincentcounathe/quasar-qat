from quasar.train.config import load_config
from quasar.train.trainer import train

if __name__ == "__main__":
    train(load_config())
