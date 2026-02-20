def crear_dataset_patron_1_9(size,length_for_seq,seed,pattern_freq=0.5, device='cpu', min_padding=0, hard_negatives = False):
    import random
    import torch
    import matplotlib.pyplot as plt
    arr_datasetx = []
    arr_datasety = [[0] for i in range(0,size)]
    lengths = []
    id_max = 12
    #CLS_ID = id_max+1
    random.seed(seed)
    for i in range(0,size):
        actual_length = random.randint(5,length_for_seq-min_padding)
        lengths.append(actual_length)
        actual_seq = []
        for j in range(0,length_for_seq):
            if(j < actual_length):
                nrandom = random.randint(1,id_max)
                # To avoid the pattern be created before the assignation of pattern_freq
                if j>0 and actual_seq[j-1]==1:
                    while(nrandom == 9):
                        nrandom = random.randint(1,id_max)
                actual_seq.append(nrandom)
            else:
                actual_seq.append(0)                
        arr_datasetx.append(actual_seq)
    pattern_positions = []
    number_of_seqs_with_patterns = int(size*pattern_freq)
    for i in range(0,number_of_seqs_with_patterns):
        while(True):
            nrandom = random.randint(0,size-1)
            if(nrandom not in pattern_positions):
                break
        pattern_positions.append(nrandom)
        random_pos = random.randint(0,lengths[nrandom]-2)    
        arr_datasetx[nrandom][random_pos] = 1
        arr_datasetx[nrandom][random_pos+1] = 9
        arr_datasety[nrandom][0] = 1
    
    # hard negatives
    if hard_negatives:
        for i in range(0,len(arr_datasetx)):
            if rule_1_9(arr_datasetx[i]):
                continue
            else:
                while(True):
                    nrandom1 = random.randint(0,lengths[i]-1)
                    if(nrandom1 == lengths[i]-1 or arr_datasetx[i][nrandom1+1] != 9):
                        break
                while(True):
                    nrandom2 = random.randint(0,lengths[i]-1)
                    if(nrandom1 != nrandom2 and nrandom2 != nrandom1 + 1 and arr_datasetx[i][nrandom2-1] != 1):
                        break
                arr_datasetx[i][nrandom1] = 1
                arr_datasetx[i][nrandom2] = 9

    # to tensor

    datasetx = torch.tensor(arr_datasetx,dtype=torch.long)
    datasety = torch.tensor(arr_datasety,dtype=torch.float)

    #N, T = datasetx.shape
    #datasetx_cls = torch.zeros(N, T+1,dtype=torch.long).to(device)

    #datasetx_cls[:,0] = CLS_ID
    #datasetx_cls[:, 1:] = datasetx

    return datasetx.to(device),datasety.to(device)

def rule_1_9(seq):
    toret = False

    if(isinstance(seq,list)):
        ssize = len(seq)-1
    else:
         ssize = seq.size(0)-1

    for i in range(0,ssize):
        if (seq[i] == 0):
            break
        elif (seq[i] == 1 and seq[i+1] == 9):
            toret = True
            break
    return toret
                
def how_many_1_and_9(data):
    cont = 0
    for seq in data:
        if ((1 in seq) and (9 in seq)):
            cont+=1
    return cont


def how_many_1_and_9_adjacents(data):
    cont = 0
    for seq in data:
        if rule_1_9(seq):
            cont+=1
    return cont