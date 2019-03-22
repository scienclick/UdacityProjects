
import argparse
import utils as utils

parser = argparse.ArgumentParser(description=' Predicts the Catagory')
# Command Line ardguments

parser.add_argument('-cdr','--load_chkpnt_dir',  action="store",default='mycheckpoint.pth',help='file that check point is saved')
parser.add_argument('-tdr','--target_folder_dir', action="store",default='flowers/test/1/image_06743.jpg',help='target file to predict')
parser.add_argument('-ep','--topk', action="store", type=int, default=5,help='Top K likely labels to be reported')
parser.add_argument('-js','--jsonfile', action="store", type=str,default='cat_to_name.json', help='Jason file for labeling')
args=parser.parse_args()

if __name__ == '__main__':
    utils.predict(args.load_chkpnt_dir, args.target_folder_dir,args.jsonfile,args.topk)