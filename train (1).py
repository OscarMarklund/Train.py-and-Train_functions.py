import argparse

parser = argparse.ArgumentParser('Arguments for the test.py file')

parser.add_argument('--arch', type=str, default='vgg16', help='Input architecture name - Model architecture of choice. Enclose in ''. Suggested models: vgg11, vgg13, vgg16, vgg19.')
parser.add_argument('--hul1', type=int, default=5000, help='Input an integer - Number of units in the first hidden layer. Suggested 20000>x>500.')
parser.add_argument('--hul2', type=int, default=1000, help='Input an integer - Number of units in the second hidden layer. Suggested 5000>x>200 or smaller than layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Input a number - Magnitude of weight change between backpasses. Suggested 0.05>x>0.0005')
parser.add_argument('--gpu', type=str, default='no', help='Input yes/no - Whether the model is pitched to the GPU or remains on the local CPU. Enclose in ''.')
parser.add_argument('--epochs', type=int, default=8, help='Input an integer - How many reruns the training completes before concluding.')
parser.add_argument('--sd', type=str, default='/home/workspace/ImageClassifier/checkpoint.pth', help="Input a directory - establishes a directory and name to save as the trained checkpoint. Suggested either the working directory for permanancy '/home/workspace/ImageClassifier/checkpoint.pth' or '/opt/checkpoint.pth' for temporary holding. Enclose directory within ''.")

args = parser.parse_args()

model_architecture = args.arch
hidden_layer_1_units = args.hul1
hidden_layer_2_units = args.hul2
learning_rate = args.lr
gpu = args.gpu
epochs = args.epochs
save_directory = args.sd


if __name__ == "__main__":
    from train_functions import trainandsave
    trainandsave().format(hidden_layer_1_units, hidden_layer_1_units, hidden_layer_2_units, hidden_layer_2_units, learning_rate, gpu, epochs, epochs, learning_rate, save_directory, save_directory)