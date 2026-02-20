import torch

class transformer_encoder(torch.nn.Module):
    def __init__(self, data_X, nheads, device):
        super().__init__()
        id_max = 13 # Number of possible token ids (1-12).
        d_model = 80 # Dimensions for each embedding.
        self.max_positions = data_X.shape[1] # Number max of position in a batch
        self.pos_ids = torch.tensor([i for i in range(0,self.max_positions)],dtype=torch.long,device=device)
        self.device = device

        # Transformer block

        self.embed = torch.nn.Embedding(num_embeddings=id_max,embedding_dim=d_model,padding_idx=0,device=device)
        self.pos_embed = torch.nn.Embedding(num_embeddings=self.max_positions,embedding_dim=d_model,device=device)
        
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model,nheads,batch_first=True,dropout=0,device=device)

        # Classiffier

        nlayers = 3
        layers = []
        neurons = d_model
        activation = torch.nn.ReLU
        output = 1

        for i in range(0,nlayers):
            layers.append(torch.nn.Linear(neurons,int(neurons/2),device=device))
            layers.append(activation())
            #layers.append(torch.nn.Dropout(0.1*(nlayers-i)))
            neurons = int(neurons/2)
        layers.append(torch.nn.Linear(neurons,output))
        
        self.clasiffier = torch.nn.Sequential(*layers).to(device)

        
        
    def forward(self,X):
        # Getting mask through length        
        T = self.max_positions
        idx = torch.arange(T,device=self.device)
        k_idx = idx.view(1,T) #Columns (1,2,3)

        # padding mask
        valid = (X != 0)
        lengths = torch.sum(valid,dim=1).to(self.device)
        arr_padding_mask = (k_idx >= lengths[:,None]) # (1,2,3,4,5) , (5,6,7,8,9)

        # Positional Encoding
        X = self.embed(X) 
        P = self.pos_embed(self.pos_ids).unsqueeze(0)
        embed = torch.add(X,P)*valid[:,:,None]
        #print(embed)

        # Transformer
        encoder_layer_result = self.encoder_layer(embed,src_key_padding_mask=arr_padding_mask)

        # Pooling
        enc_sum = torch.sum(encoder_layer_result*valid[:,:,None],dim=1) 
        avg = torch.div(enc_sum,lengths[:,None])

        #Classifier
        logits = self.clasiffier(avg).to(self.device) 
        #print("logits." , logits)
        return logits