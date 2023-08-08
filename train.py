# Python program to demonstrate
# command line arguments

import argparse
import os,sys


# Get the current working directory (assuming main.py is located in the 'project' folder)
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the 'utils' directory path to the Python path
utils_path = os.path.join(current_path, "Assignment 1")
sys.path.append(utils_path)



import backprop

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-wp", "--wandb_project", help = "Project name used to track experiments in Weights & Biases dashboard", default= "myprojectname")
parser.add_argument("-we", "--wandb_entity", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.", default= "myname")
parser.add_argument("-d", "--dataset", help="choices: [mnist, fashion_mnist]", default= "fashion_mnist")
parser.add_argument("-b", "--batch_size", help="Batch size used to train neural network.", default= "4")
parser.add_argument("-e", "--epochs", help="Number of epochs to train neural network", default= "1")
parser.add_argument("-l", "--loss", help="Mention loss function name : choices:  [mean_squared_error, cross_entropy]", default= "cross_entropy")
parser.add_argument("-o", "--optimizer", help="choices: [sgd, momentum, nag, rmsprop, adam, nadam]", default= "sgd")
parser.add_argument("-lr", "--learning_rate", help="Learning rate used to optimize model parameters", default="0.1")
parser.add_argument("-m", "--momentum", help = "Momentum used by momentum and nag optimizers.", default="0.5")
parser.add_argument("-beta", "--beta", help="Beta used by rmsprop optimizer", default="0.5")
parser.add_argument("-beta1", "--beta1", help="Beta1 used by adam and nadam optimizers.", default="0.5")
parser.add_argument("-beta2", "--beta2", help="Beta2 used by adam and nadam optimizers.", default="0.5")
parser.add_argument("-eps", "--epsilon", help="Epsilon used by optimizers.", default="0.000001")
parser.add_argument("-w_d", "--weight_decay", help="Weight decay used by optimizers.", default=".0")
parser.add_argument("-w_i", "--weight_init", help="choices: [random, Xavier]", default="random")
parser.add_argument("-nhl", "--num_layers", help="Number of hidden layers used in feedforward neural network.", default="1")
parser.add_argument("-sz", "--hidden_size", help="Number of hidden neurons in a feedforward layer.", default="4")
parser.add_argument("-a", "--activation", help="choices: [identity, sigmoid, tanh, ReLU]", default="sigmoid")

# Read arguments from command line
args = parser.parse_args()
argstore = {}

argstore.update({"wandb_project": args.wandb_project})
argstore.update({"wandb_entity": args.wandb_entity})
argstore.update({"dataset": args.dataset})
argstore.update({"epochs": args.epochs})
argstore.update({"batch_size": args.batch_size})
argstore.update({"loss": args.loss})
argstore.update({"optimizer": args.optimizer})
argstore.update({"learning_rate": args.learning_rate})
argstore.update({"momentum": args.momentum})
argstore.update({"beta": args.beta})
argstore.update({"beta1": args.beta1})
argstore.update({"beta2": args.beta2})
argstore.update({"epsilon": args.epsilon})
argstore.update({"weight_decay": args.weight_decay})
argstore.update({"weight_init": args.weight_init})
argstore.update({"num_layers": args.num_layers})
argstore.update({"hidden_size": args.hidden_size})
argstore.update({"activation": args.activation})


print("parameters: ")
print(argstore)
backprop.SimpleClassifier(argstore)

