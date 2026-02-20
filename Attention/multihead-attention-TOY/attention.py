import torch, math
# Q, K, V: (B, h, T, d)
# scores = Q @ Kᵀ`: (B, h, T, T) con Kᵀ = K.transpose(-2, -1) → (B, d, T)
# weights = softmax(scores, dim=-1) → softmax over keys (última dim)
# out = weights @ V → (B, h, T, d)
# Q = ROWS || K = COLUMNS
def forward_att(Q,K,V,mask = False,length=None):
    if length == None:
        length = Q.shape[-2]
    if not (Q.shape == K.shape and Q.shape[0:2] == V.shape[0:2]):
        raise ValueError("Size of tensors Q and K isn't equivalent")
    if any(length <= 0):
        raise ValueError("Length can't be lower or equal to 0")
    tK = K.transpose(-2,-1)
    scores = Q @ tK # shape -> ([1, 2, 3, 3])
    if mask:
        T = scores.shape[-1]
        idx = torch.arange(T, device=scores.device)
        q_idx = idx.view(1,1,T,1) #Rows   (1,2,3)
        k_idx = idx.view(1,1,1,T) #Columns (1,2,3)
        arr_causal_mask = k_idx > q_idx
        # padding mask
        length = length.view(1,1,length.shape[-1],1)
        
        arr_padding_mask = k_idx >= length
        arr_padding_mask = arr_padding_mask.permute(dims=(2,1,0,3)) # (0,1,2,3) -> (2,1,0,3)
        arr_padding_mask = arr_padding_mask.expand((-1,-1,scores.shape[-1],-1))
        arr_mask = arr_causal_mask | arr_padding_mask
    else:
        arr_mask = torch.zeros(scores.shape,dtype=torch.bool)
    scores = scores/math.sqrt(K.shape[-1])
    scores_masked = scores.masked_fill(arr_mask,float('-inf'))
    weights = torch.softmax(scores_masked, dim =-1)
    out = weights @ V
    return out

def main():
    # dim => (B,h,T,d) => (B,T,h*d)
    h1 = [[1,-2],[-5,6],[-3,4]]
    h2 = [[-1,3],[4,-1],[2,-4]]
    h3 = [[2,-1],[-3,4],[-5,1]]
    h4 = [[0,2],[5,3],[2,-2]]
    Q = torch.tensor([[h1,h2],[h3,h4]],dtype=torch.float64) 
    K = torch.tensor([[h1,h2],[h3,h4]],dtype=torch.float64)
    V = torch.tensor([[h1,h2],[h3,h4]],dtype=torch.float64)
    batch1 = torch.tensor([h1,h2],dtype=torch.float64)
    batch2 = torch.tensor([h3,h4],dtype=torch.float64)

    length = torch.tensor([2,1],dtype=torch.int64)
    out = forward_att(Q,K,V,mask = True,length=length)

    out_permuted = out.permute(dims=(0,2,1,3)) #(2,2,3,2) -> (2,3,2,2)
    out_reshaped = out_permuted.reshape((out_permuted.shape[0],out_permuted.shape[1],-1)) #(2,3,4)
    hn = torch.stack((batch1,batch2),dim=0) # (2,2,3,2)
    hn_permuted = hn.permute(dims=(0,2,1,3)) # (2,2,3,2) -> (2,3,2,2)
    hn_reshaped = hn_permuted.reshape((hn_permuted.shape[0],hn_permuted.shape[1],-1)) #(2,3,4)
    result = out_reshaped + hn_reshaped
    print(result)

if __name__ == "__main__":
    main()




    

