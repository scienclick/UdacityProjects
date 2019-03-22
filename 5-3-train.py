
import argparse
import utils as utils

parser = argparse.ArgumentParser(description=' Trains the Model')
# Command Line ardguments

parser.add_argument('-srd','--sourcedata_dir', nargs='*', action="store", default="flowers/",help='Directory containing files')
parser.add_argument('-hu','--hidden_layers', type=int, action="store", default=500,help='Hidden layers')
parser.add_argument('-arch','--arch',action="store", default='vgg16',type=str,help='vgg16,densenet121')
parser.add_argument('-pu','--pu', action="store", default="gpu",help='cpu or gpu')
parser.add_argument('-svd','--save_dir', action="store", default="mycheckpoint.pth",help='file name to save check point')
parser.add_argument('-lr','--learning_rate', action="store", default=0.001,help='learning rate')
parser.add_argument('-dr','--dropout', action = "store", default = 0.5,help='Drop out')
parser.add_argument('-ep','--epochs', action="store", type=int, default=1,help='Epochs')



args=parser.parse_args()

if __name__ == '__main__':
    train_data,trainloader,validloader,testloader= utils.loader_data(args.sourcedata_dir)
    model= utils.setupNN(args.dropout, args.hidden_layers,args.arch)
    utils.RunSaveNN(model, args.pu, args.learning_rate, args.epochs, trainloader, validloader, args.hidden_layers, args.save_dir,args.arch,train_data)