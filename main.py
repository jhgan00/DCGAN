from model import DCGAN
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', type=int, help='train epochs')

args = parser.parse_args()

model = DCGAN()
model.train(args.epochs)