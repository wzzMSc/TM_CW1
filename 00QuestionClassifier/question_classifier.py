import argparse
from train import Train
from test import Test

parser = argparse.ArgumentParser(description="Text Mining Course Work 1")
parser.add_argument('phase', type=str, help="Phase of question classification.(train/test)")
parser.add_argument('-train_file',type=str,help="training_file_path")
parser.add_argument('-model',type=str,help="bow/bilstm")
parser.add_argument('-config_file',type=str,help="configuration_file_path")
parser.add_argument('-model_file',type=str,help="model_path")
parser.add_argument('-test_file',type=str,help="test_file_path")
args = parser.parse_args()

if args.phase == 'train':
    Train(args.train_file,args.model,args.config_file,args.model_file).train()
if args.phase == 'test':
    Test(args.model_file,args.model,args.test_file)