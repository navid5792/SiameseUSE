# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import time
import argparse
import pickle
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from data import get_nli, get_batch, build_vocab
from mutils import get_optimizer
from models import NLINet
from read_data import get_MSRP_data, get_SICK_data, get_AI_data, get_SICK_tree_data, get_QQP_data, get_SICK_binary_data

filename = "Results/infersent_cic_MSRP"
print(filename, "Branch attention")
parser = argparse.ArgumentParser(description='NLI training')
# paths
parser.add_argument("--nlipath", type=str, default='dataset/SNLI/', help="NLI data path (SNLI or MultiNLI)")
parser.add_argument("--outputdir", type=str, default='savedir/', help="Output directory")
parser.add_argument("--outputmodelname", type=str, default='model_cic_SICK_infersent.pickle')
parser.add_argument("--word_emb_path", type=str, default="dataset/glove.840B.300d.txt", help="word embedding file path") # glove.840B.300d.txt

# training
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dpout_model", type=float, default=0.1, help="encoder dropout")
parser.add_argument("--dpout_fc", type=float, default=0.1, help="classifier dropout")
parser.add_argument("--nonlinear_fc", type=float, default=5, help="use nonlinearity in fc")
parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1", help="adam or sgd,lr=0.1")
parser.add_argument("--lrshrink", type=float, default=1, help="shrink factor for sgd")
parser.add_argument("--decay", type=float, default=0.99, help="lr decay")
parser.add_argument("--minlr", type=float, default=1e-5, help="minimum lr")
parser.add_argument("--max_norm", type=float, default=5., help="max norm (grad clipping)")

# model
parser.add_argument("--encoder_type", type=str, default='InferSent', help="see list of encoders")
parser.add_argument("--enc_lstm_dim", type=int, default=600, help="encoder nhid dimension")
parser.add_argument("--n_enc_layers", type=int, default=1, help="encoder num layers")
parser.add_argument("--fc_dim", type=int, default=150, help="nhid of fc layers")
parser.add_argument("--n_classes", type=int, default=3, help="entailment/neutral/contradiction")
parser.add_argument("--pool_type", type=str, default='max', help="max or mean")

# gpu
parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
parser.add_argument("--seed", type=int, default=1234, help="seed")

# data
parser.add_argument("--word_emb_dim", type=int, default=300, help="word embedding dimension")

params, _ = parser.parse_known_args()

# set gpu device
torch.cuda.set_device(params.gpu_id)

# print parameters passed, and all parameters
print('\ntogrep : {0}\n'.format(sys.argv[1:]))
print(params)

"""
SEED
"""
np.random.seed(params.seed)
torch.manual_seed(params.seed)
torch.cuda.manual_seed(params.seed)

"""
DATA
"""
#train, test, valid = get_nli(params.nlipath)

train, valid, test = get_SICK_data()


word_vec = build_vocab(train['s1'] + train['s2'] +
                       valid['s1'] + valid['s2'] +
                       test['s1'] + test['s2'], params.word_emb_path)

s = np.random.uniform(low=-0.5, high=0.5, size=(300,))
s1 = np.random.uniform(low=-0.5, high=0.5, size=(300,))
number = np.random.uniform(low=-0.5, high=0.5, size=(300,))
word_vec['<s>'] = s
word_vec['</s>'] = s1
word_vec['<number>'] = number

for split in ['s1', 's2']:
    for data_type in ['train', 'valid', 'test']:
        eval(data_type)[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] +
                                                                     ['</s>'] for sent in eval(data_type)[split]])

parser = argparse.ArgumentParser(description='Training Hyperparams')
# data loading params
parser.add_argument('-data_path', default = "data")

# network params
parser.add_argument('-d_model', type=int, default=300)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)
parser.add_argument('-d_ff', type=int, default=2048)
parser.add_argument('-n_heads', type=int, default=1)
parser.add_argument('-n_layers', type=int, default=2)
parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-share_proj_weight', action='store_true')
parser.add_argument('-share_embs_weight', action='store_true')
parser.add_argument('-weighted_model', action='store_true') 

# training params
parser.add_argument('-lr', type=float, default=0.0002)
parser.add_argument('-max_epochs', type=int, default=10)
parser.add_argument('-batch_size', type=int, default=128)
parser.add_argument('-max_src_seq_len', type=int, default=300)
parser.add_argument('-max_tgt_seq_len', type=int, default=300)
parser.add_argument('-max_grad_norm', type=float, default=None)
parser.add_argument('-n_warmup_steps', type=int, default=4000)
parser.add_argument('-display_freq', type=int, default=100)
parser.add_argument('-src_vocab_size', type=int, default=len(word_vec))
parser.add_argument('-tgt_vocab_size', type=int, default=len(word_vec))
parser.add_argument('-log', default=None)
parser.add_argument('-model_path', type=str, default = "")

transformer_opt = parser.parse_args()

"""
MODEL
"""
# model config
config_nli_model = {
    'n_words'        :  len(word_vec)          ,
    'word_emb_dim'   :  params.word_emb_dim   ,
    'enc_lstm_dim'   :  params.enc_lstm_dim   ,
    'n_enc_layers'   :  params.n_enc_layers   ,
    'dpout_model'    :  params.dpout_model    ,
    'dpout_fc'       :  params.dpout_fc       ,
    'fc_dim'         :  params.fc_dim         ,
    'bsize'          :  params.batch_size     ,
    'n_classes'      :  params.n_classes      ,
    'pool_type'      :  params.pool_type      ,
    'nonlinear_fc'   :  params.nonlinear_fc   ,
    'encoder_type'   :  params.encoder_type   ,
    'use_cuda'       :  True                  ,

}

# model
encoder_types = ['InferSent', 'BLSTMprojEncoder', 'BGRUlastEncoder',
                 'InnerAttentionMILAEncoder', 'InnerAttentionYANGEncoder',
                 'InnerAttentionNAACLEncoder', 'ConvNetEncoder', 'LSTMEncoder']
assert params.encoder_type in encoder_types, "encoder_type must be in " + \
                                             str(encoder_types)
nli_net = NLINet(config_nli_model, transformer_opt)
print(nli_net)
#nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))
#print("Model Loaded")


# loss
weight = torch.FloatTensor(params.n_classes).fill_(1)
loss_fn = nn.CrossEntropyLoss(weight=weight)
loss_fn.size_average = False

# optimizer
optim_fn, optim_params = get_optimizer(params.optimizer)
optimizer = optim_fn(nli_net.parameters(), **optim_params)

# cuda by default
nli_net.cuda()
nli_net.filter_parameters()
loss_fn.cuda()

for name, x in nli_net.named_parameters():
    print(name)

"""
TRAIN
"""
val_acc_best = -1e10
adam_stop = False
stop_training = False
lr = optim_params['lr'] if 'sgd' in params.optimizer else None


def trainepoch(epoch):
    print('\nTRAINING : Epoch ' + str(epoch))
    nli_net.train()
    all_costs = []
    logs = []
    words_count = 0

    last_time = time.time()
    correct = 0.
    # shuffle the data
    permutation = np.random.permutation(len(train['s1']))
    #permutation = np.arange(len(train['s1']))

    s1 = train['s1'][permutation]
    s2 = train['s2'][permutation]
    target = train['label'][permutation]


    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * params.decay if epoch>1\
        and 'sgd' in params.optimizer else optimizer.param_groups[0]['lr']
    print('Learning rate : {0}'.format(optimizer.param_groups[0]['lr']))

    for stidx in range(0, len(s1), params.batch_size):
        # prepare batch

        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size],
                                     word_vec, params.word_emb_dim)       
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())

        #print(s1_batch.size(), "\n\n", s2_batch.size(), s1_len)
        #asaas
        #print("\n\n")
        #print(s1_batch, s2_batch)
        

        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        k = s1_batch.size(1)  # actual batch size
        
        s1_batch = s1_batch.transpose(0,1)
        s2_batch = s2_batch.transpose(0,1)

        position_a = torch.zeros(s1_batch.size(0), s1_batch.size(1)) # 16 21
        mask_a = torch.zeros(s1_batch.size(0), s1_batch.size(1))
        for i in range(s1_batch.size(0)): # 16
            j = 1
            torch.fill_(position_a[i], int(s1_len[i]))
            for k in range(s1_len[i]): #21
                position_a[i][k] = j
                j += 1
                mask_a[i][k] = 1
        position_a = position_a.long()

        position_b = torch.zeros(s2_batch.size(0), s2_batch.size(1)) # 16 21
        mask_b = torch.zeros(s2_batch.size(0), s2_batch.size(1))
        for i in range(s2_batch.size(0)): # 16
            j = 1
            torch.fill_(position_b[i], int(s2_len[i]))
            for k in range(s2_len[i]): #21
                position_b[i][k] = j
                j += 1
                mask_b[i][k] = 1
        position_b = position_b.long()
        

        position_a, position_b = Variable(position_a.cuda()), Variable(position_b.cuda())
        # model forward
        output = nli_net((s1_batch, s1_len, position_a, mask_a.cuda()), (s2_batch, s2_len, position_b , mask_b.cuda()))
        #print(output, tgt_batch)
        #print(nli_net.classifier[1].bias)
        #print(output)
        #asasas
        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()
        assert len(pred) == len(s1[stidx:stidx + params.batch_size])

        # loss

        loss = loss_fn(output, tgt_batch)
        all_costs.append(float(loss))
        words_count += (s1_batch.nelement() + s2_batch.nelement()) / params.word_emb_dim

        # backward
        optimizer.zero_grad()
        loss.backward()
        #print(nli_net.encoder.enc_lstm.bias_ih_l0.grad)

        # gradient clipping (off by default)
        shrink_factor = 1
        total_norm = 0
        k = s1_batch.size(0)
        count = 0
        for name,p in nli_net.named_parameters():
            #print(name)
            #print(total_norm)
            if p.requires_grad:
                count += 1
#                print(name)
                p.grad.data.div_(k)  # divide by the actual batch size
                total_norm += p.grad.data.norm() ** 2
#        print("\n\n\n\n")
        #print(nli_net.classifier[1].bias, nli_net.encoder.enc_lstm.weight_hh_l0, k)
        #print(total_norm)
        #asasas
        total_norm = np.sqrt(total_norm)

        if total_norm > params.max_norm:
            print("shrinking applied...................")
            shrink_factor = params.max_norm / total_norm
        current_lr = optimizer.param_groups[0]['lr'] # current lr (no external "lr", for adam)
        optimizer.param_groups[0]['lr'] = current_lr * shrink_factor # just for update

        # optimizer step
        optimizer.step()
        optimizer.param_groups[0]['lr'] = current_lr
        if len(all_costs) == 100:
            logs.append('{0} ; loss {1} ; sentence/s {2} ; words/s {3} ; accuracy train : {4}'.format(
                            stidx, round(np.mean(all_costs), 2),
                            int(len(all_costs) * params.batch_size / (time.time() - last_time)),
                            int(words_count * 1.0 / (time.time() - last_time)),
                            round(100.* int(correct.data)/(stidx+k), 2)))
            print(logs[-1])
            last_time = time.time()
            words_count = 0
            all_costs = []
    train_acc = round(100 * int(correct.data)/len(s1), 2)
    print('results : epoch {0} ; mean accuracy train : {1}'
          .format(epoch, train_acc))
    return train_acc


def evaluate(epoch, eval_type='valid', final_eval=False):
    nli_net.eval()
    correct = 0.
    global val_acc_best, lr, stop_training, adam_stop

    if eval_type == 'valid':
        print('\nVALIDATION : Epoch {0}'.format(epoch))

    s1 = valid['s1'] if eval_type == 'valid' else test['s1']
    s2 = valid['s2'] if eval_type == 'valid' else test['s2']
    target = valid['label'] if eval_type == 'valid' else test['label']

    for stidx in range(0, len(s1), params.batch_size):
        #print(stidx, "/", len(s1))
        # prepare batch

        #print(s1[i], "\n", s2[i], s1[i:i + params.batch_size])
        #print(s1[stidx], "\n", s2[stidx])
        #print(s1[stidx:stidx + params.batch_size],"\n\n\n", s2[stidx:stidx + params.batch_size], "\n\n\n", target[stidx:stidx + params.batch_size], "\n\n\n")
        s1_batch, s1_len = get_batch(s1[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim)
        s2_batch, s2_len = get_batch(s2[stidx:stidx + params.batch_size], word_vec, params.word_emb_dim)
        s1_batch, s2_batch = Variable(s1_batch.cuda()), Variable(s2_batch.cuda())
        tgt_batch = Variable(torch.LongTensor(target[stidx:stidx + params.batch_size])).cuda()
        
        s1_batch = s1_batch.transpose(0,1)
        s2_batch = s2_batch.transpose(0,1)

        position_a = torch.zeros(s1_batch.size(0), s1_batch.size(1)) # 16 21
        mask_a = torch.zeros(s1_batch.size(0), s1_batch.size(1))
        for i in range(s1_batch.size(0)): # 16
            j = 1
            torch.fill_(position_a[i], int(s1_len[i]))
            for k in range(s1_len[i]): #21
                position_a[i][k] = j
                j += 1
                mask_a[i][k] = 1
        position_a = position_a.long()

        position_b = torch.zeros(s2_batch.size(0), s2_batch.size(1)) # 16 21
        mask_b = torch.zeros(s2_batch.size(0), s2_batch.size(1))
        for i in range(s2_batch.size(0)): # 16
            j = 1
            torch.fill_(position_b[i], int(s2_len[i]))
            for k in range(s2_len[i]): #21
                position_b[i][k] = j
                j += 1
                mask_b[i][k] = 1
        position_b = position_b.long()
        
        position_a, position_b = Variable(position_a.cuda()), Variable(position_b.cuda())
        # model forward
        output = nli_net((s1_batch, s1_len, position_a, mask_a.cuda()), (s2_batch, s2_len, position_b, mask_b.cuda()))

        pred = output.data.max(1)[1]
        correct += pred.long().eq(tgt_batch.data.long()).cpu().sum()

    # save model
    eval_acc = round(100 * int(correct.data) / len(s1), 2)
    if final_eval:
        print('finalgrep : accuracy {0} : {1}'.format(eval_type, eval_acc))
    else:
        print('togrep : results : epoch {0} ; mean accuracy {1} :\
              {2}'.format(epoch, eval_type, eval_acc))
        with open(filename, "a") as f:
            f.write(str(epoch)+ " " + str(eval_acc) + "\n")

    if eval_type == 'valid' and epoch <= params.n_epochs:
        if eval_acc > val_acc_best:
            print('saving model at epoch {0}'.format(epoch))
            if not os.path.exists(params.outputdir):
                os.makedirs(params.outputdir)
            torch.save(nli_net.state_dict(), os.path.join(params.outputdir,
                       params.outputmodelname))
            val_acc_best = eval_acc
        else:
            if 'sgd' in params.optimizer:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / params.lrshrink
                print('Shrinking lr by : {0}. New lr = {1}'
                      .format(params.lrshrink,
                              optimizer.param_groups[0]['lr']))
                if optimizer.param_groups[0]['lr'] < params.minlr:
                    stop_training = True
            if 'adam' in params.optimizer:
                # early stopping (at 2nd decrease in accuracy)
                stop_training = adam_stop
                adam_stop = True
    return eval_acc


"""
Train model on Natural Language Inference task
"""
epoch = 1

while not stop_training and epoch <= params.n_epochs:
    train_acc = trainepoch(epoch)
    eval_acc = evaluate(epoch, 'valid')
    epoch += 1

# Run best model on test set.
nli_net.load_state_dict(torch.load(os.path.join(params.outputdir, params.outputmodelname)))

print('\nTEST : Epoch {0}'.format(epoch))
evaluate(1e6, 'test', True)
#evaluate(0, 'test', True)

# Save encoder instead of full model
#torch.save(nli_net.encoder.state_dict(), os.path.join(params.outputdir, params.outputmodelname + '.encoder.pkl'))
