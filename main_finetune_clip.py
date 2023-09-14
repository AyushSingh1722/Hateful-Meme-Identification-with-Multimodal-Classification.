from custom_collate import collate_mismatch
import dataset_mismatch  
import json 
import os 
import torch
import torchvision
import torch.nn as nn
import numpy as np 
from torch.utils.data import DataLoader
import argparse
import io
import clip_classifier
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import time 
import clip 
from sklearn.metrics import confusion_matrix, classification_report
import pickle
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts


parser = argparse.ArgumentParser(description='Training using the precomputed embeddings')
##### locations #####  
              
parser.add_argument('--exp_folder', type=str, default='./exp/',
                    help='path to the folder to log the output and save the models')
                    
###### model details ########                    
parser.add_argument('--pdrop', type=float, default=0.5,
                    help='dropout probability')


##### Training details #####
parser.add_argument('--batch_size', type=int, default=32,
                    help='dimension of domains embeddings') 
parser.add_argument('--num_workers', type=int, default=6,
                    help='number of data loaders workers') 
parser.add_argument('--epochs', type=int, default = 30,
                    help='number of epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
parser.add_argument('--log_interval', type=int, default=200,
                    help='how many batches')

                    # /mnt/c61a35bf-fc59-4aab-a996-b254f9ab9052/Suraj/OPENAI_CLIP/finetuned_clip/best_model_acc.pth.tar
parser.add_argument('--resume', type=str, default = '', help='path to model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='which optimizer to use')
parser.add_argument('--lr_clip', type=float, default=5e-5,
                    help='learning rate of the clip model')
parser.add_argument('--lr_classifier', type=float, default=5e-4,
                    help='learning rate of the clip model')                    
parser.add_argument('--sgd_momentum', type=float, default=0.9,
                    help='momentum when using sgd')                      
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--wdecay', default=1.2e-6, type=float,
                        help='weight decay pow (default: -5)')
                    
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

#### load input files ####



data_paths_hm_train = pickle.load(open('/data/home/rahulboipai/AIP_Project/CLIP_finetune/data_files/train_data_paths.pkl','rb'))
data_paths_hm_test =  pickle.load(open('/data/home/rahulboipai/AIP_Project/CLIP_finetune/data_files/test_data_paths.pkl','rb'))

device = torch.device('cuda:2')

print(device,'\n')

#### settings of the model ####
model_settings = {'pdrop': args.pdrop}
base_clip, preprocess = clip.load("ViT-B/32", device= device)
classifier_clip = clip_classifier.ClipClassifier(model_settings,base_clip)
classifier_clip.cuda()

if not os.path.exists('runs'):
    os.makedirs('runs')


log_dir  = 'runs'
logger = SummaryWriter(comment='clip_hatememe')


#### load Datasets and DataLoader ####

######## Few samples #########

train_dataset = dataset_mismatch.HateClipDataset(data_paths_hm_train, preprocess)
test_dataset = dataset_mismatch.HateClipDataset(data_paths_hm_test , preprocess)

print('train',len(data_paths_hm_train))
print('test',len(data_paths_hm_test))


# print("before_train_dataloader")
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,collate_fn = collate_mismatch, num_workers=args.num_workers, pin_memory=True)
# print("after_train_dataloader")
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = collate_mismatch,num_workers=args.num_workers,  pin_memory=True)
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = collate_mismatch, num_workers=args.num_workers,  pin_memory=True)


#create exp folder if it has not been created 
if not os.path.isdir(args.exp_folder):
    os.makedirs(args.exp_folder)

#resume training
stored_loss = 100000000   
stored_acc = 0

classifier_list = ['classifier.weight', 'classifier.bias']
classifier_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in classifier_list, classifier_clip.named_parameters()))))
base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in classifier_list, classifier_clip.named_parameters()))))

if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD([{'params': base_params}, {'params': classifier_params, 'lr': args.lr_classifier}], lr=args.lr_clip, weight_decay=args.wdecay, momentum=args.sgd_momentum)    
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam([{'params': base_params}, {'params': classifier_params, 'lr': args.lr_classifier}], lr=args.lr_clip, weight_decay=args.wdecay) 

if args.resume:
    if os.path.isfile(args.resume):
        log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'a') 
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        if 'best_val_loss' in checkpoint: stored_loss = checkpoint['best_val_loss']
        if 'best_val_acc' in checkpoint: stored_acc = checkpoint['best_val_acc']
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    log_file_val_loss = open(os.path.join(args.exp_folder,'log_file_val.txt'),'w') 
#define loss function
criterion = nn.BCEWithLogitsLoss().cuda()



params = list(classifier_clip.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


def evaluate(loader):
    total_loss = 0
    correct = 0
    total_num = 0
    classifier_clip.eval()

    total_preds = []
    total_truth = []

    for (idx, batch) in enumerate(loader):
        labels = batch[0].cuda()
        imgs = batch[1].cuda() 
        captions = batch[2].cuda()
    
        with torch.no_grad():
            #forward 
            output = classifier_clip(imgs,captions) 
            #compute loss     
            loss = criterion(output, torch.unsqueeze(labels.float(), 1))
            total_loss += loss.item()
            #compute correct predictions 
            pred = torch.sigmoid(output) >= 0.5
            truth = torch.unsqueeze(labels,1) >= 0.5
            pred_m = pred.cpu()
            truth_m = truth.cpu()
            total_preds.append(np.array(pred_m))
            total_truth.append(np.array(truth_m))
            # print(truth)
            correct += pred.eq(truth).sum().item()
            total_num += labels.size(0) 
    avg_loss = total_loss/len(loader)    
    acc = (correct/total_num)*100
    return avg_loss, acc , total_preds , total_truth

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None: p.grad.data = p.grad.data.float() 
    
def train():
    total_loss = 0
    start_time = time.time()

    classifier_clip.train()

    for (idx, batch) in enumerate(train_dataloader):

        labels = batch[0].cuda()
        imgs = batch[1].cuda() 
        captions = batch[2].cuda()

        #forward 
        output = classifier_clip(imgs,captions) 
        #compute loss     
        loss = criterion(output, torch.unsqueeze(labels.float(), 1))
        total_loss += loss.item()
    
        #backward and optimizer step 
        optimizer.zero_grad()
        loss.backward()
        convert_models_to_fp32(classifier_clip)
        optimizer.step()
        clip.model.convert_weights(classifier_clip)
        #log    
        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.8f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                    epoch, idx, len(train_dataloader) , optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
            
     
try:

    for param in classifier_clip.parameters():
            param.requires_grad = False

    
    for param in classifier_clip.classifier.parameters():
            param.requires_grad = True


    for param in classifier_clip.clip.ln_final.parameters():
                param.requires_grad = True

    for epoch in range(args.start_epoch, args.epochs+1):
        epoch_start_time = time.time()

        train()
        # print("Line 202")

        train_loss, train_acc,_,_ = evaluate(train_dataloader)
        # val_loss, val_acc,_,_ = evaluate(val_dataloader)
        
    
        logger.add_scalar(log_dir + '/train-loss', train_loss, epoch)
        logger.add_scalar(log_dir + '/train-acc', train_acc, epoch)
        
        print('-' * 89)
        print('| train loss {:5.2f} | train acc {:8.3f}'.format(train_loss, train_acc))
        print('| end of epoch {:3d} | time: {:5.2f}s |'.format(
                    epoch, (time.time() - epoch_start_time)))
        print('-' * 89)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
    


# Load the best saved model if exists and run for the test data.
test_loss, test_acc, total_predictions, total_truths = evaluate(test_dataloader)

tot_preds= []
for ele in total_predictions :
    for e in ele:
        tot_preds.append(int(e))

tot_truths= []
for ele in total_truths :
    for e in ele :
        tot_truths.append(int(e))    


confusion_matrix(tot_truths, tot_preds)
print('=' * 89)
print('=' * 89)
print(classification_report(tot_truths, tot_preds))
print("no_of_epochs:{} , lr_clip:{}  , lr_classifier:{} , pdrop:{}" .format(args.epochs,args.lr_clip,args.lr_classifier,args.pdrop))
print('=' * 89)
print('| End of training |  best acc | test loss {:5.2f} | test acc {:8.3f}'.format(
test_loss, test_acc))
print('=' * 89)


