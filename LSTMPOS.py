import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq,to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"DET":0,"NN":1,"V":2}
char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        for ch in word:
            if ch not in char_to_ix:
                char_to_ix[ch] = len(char_to_ix)

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
HIDDEN_CHAR_DIM = 4
CHAR_DIM = 4
EPOCHS = 100
class charLSTM(nn.Module):
    def __init__(self,char_emb_dim,hidden_dim,char_size):
        super(charLSTM,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(char_size,char_emb_dim)
        self.lstm = nn.LSTM(char_emb_dim,hidden_dim)
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))
    def forward(self,token):
        embed = self.word_embeddings(token)
        lstm_out,self.hidden = self.lstm(embed.view((len(token),1,-1)),self.hidden)

        return lstm_out[-1]

    

    


class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,target_size):
        super(LSTMTagger,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim,target_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1,1,self.hidden_dim),torch.zeros(1,1,self.hidden_dim))

    def forward(self,sentence,char_rep_sent):
        embed = self.word_embeddings(sentence).view(len(sentence),1,-1)
        combine = torch.cat((embed,char_rep_sent),2)
        lstm_out, self.hidden = self.lstm(embed,self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_score = F.log_softmax(tag_space,dim = 1)
        return tag_score

loss_func = nn.NLLLoss()
char_model = charLSTM(CHAR_DIM,HIDDEN_CHAR_DIM,len(char_to_ix))
model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_ix),len(tag_to_ix))
optimizer = optim.SGD(model.parameters(),lr = 0.1)
with torch.no_grad():
    char_rep_words = []                    # A list that contains char representation of words
    for word in training_data[0][0]:
        char_model.hidden = char_model.init_hidden()
        char_rep = char_model(prepare_sequence(word,char_to_ix))
        char_rep_words.append(char_rep)
    char_rep_sent = torch.stack(char_rep_words)
    inputs = prepare_sequence(training_data[0][0],word_to_ix)
    print(inputs)
    tag_scores = model(inputs,char_rep_sent)
    print(tag_scores)
losses = []
for epoch_no in range(EPOCHS):
    for sentence,tags in training_data:
        model.zero_grad()
        #Set the gradient zero for all the model parameters
        model.hidden = model.init_hidden()
        char_rep_words = []                    # A list that contains char representation of words
        for word in sentence:
            char_model.hidden = char_model.init_hidden()
            char_rep = char_model(prepare_sequence(word,char_to_ix))
            char_rep_words.append(char_rep)
        char_rep_sent = torch.stack(char_rep_words)
        #print(char_rep_sent.size())

        tag_scores = model(prepare_sequence(sentence,word_to_ix),char_rep_sent)
        #Calculate the log softmax for each tag
        loss = loss_func(tag_scores,prepare_sequence(tags,tag_to_ix))
        #Compute the loss
        loss.backward()
        #Compute the gradient wrt each parameter
        optimizer.step()

with torch.no_grad():
    char_rep_words = []                    # A list that contains char representation of words
    for word in training_data[0][0]:
        char_model.hidden = char_model.init_hidden()
        char_rep = char_model(prepare_sequence(word,char_to_ix))
        char_rep_words.append(char_rep)
    char_rep_sent = torch.stack(char_rep_words)
    inputs = prepare_sequence(training_data[0][0],word_to_ix)
    print(inputs)
    tag_scores = model(inputs,char_rep_sent)
    print(tag_scores)