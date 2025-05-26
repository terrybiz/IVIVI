import torch.nn as nn
import torch
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class Embedding_Net(nn.Module):
    def __init__(self, opt):
        super(Embedding_Net, self).__init__()

        self.fc1 = nn.Linear(opt.resSize, opt.embedSize)
        self.fc2 = nn.Linear(opt.embedSize, opt.outzSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, features):
        embedding= self.relu(self.fc1(features))
        out_z = F.normalize(self.fc2(embedding), dim=1)
        return embedding,out_z
############################################################################        
class CrossAttention_MLP_G(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.noise_proj = nn.Linear(opt.nz, opt.ngh)
        self.att_proj = nn.Linear(opt.attSize, opt.ngh)
        self.attn = nn.MultiheadAttention(opt.ngh, num_heads=4, batch_first=True)
        self.fc = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)
    def forward(self, noise, att):
        n = self.lrelu(self.noise_proj(noise)).unsqueeze(1)  # [batch, 1, ngh]
        a = self.lrelu(self.att_proj(att)).unsqueeze(1)      # [batch, 1, ngh]
        # cross attention: noise queries att (可以反過來)
        out, _ = self.attn(n, a, a)                         # [batch, 1, ngh]
        out = out.squeeze(1)
        out = self.relu(self.fc(out))
        return out
######################################################################

class FiLM_MLP_G(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc_noise = nn.Linear(opt.nz, opt.ngh)
        self.fc_att_gamma = nn.Linear(opt.attSize, opt.ngh)
        self.fc_att_beta = nn.Linear(opt.attSize, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.relu = nn.ReLU(True)
        self.ln = nn.LayerNorm(opt.ngh)
        self.dropout = nn.Dropout(0.3)
        self.apply(weights_init)
    def forward(self, noise, att):
        h = self.relu(self.fc_noise(noise))
        gamma = self.fc_att_gamma(att)
        beta = self.fc_att_beta(att)
        h = gamma * h + beta
        h = self.ln(self.relu(self.fc2(h)))
        h = self.dropout(h)
        h = self.fc3(h)
        return h
#########################################################
class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h
####################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.ln1 = nn.LayerNorm(size)
        self.act1 = nn.LeakyReLU(0.2, True)
        self.fc2 = nn.Linear(size, size)
        self.ln2 = nn.LayerNorm(size)
        self.dropout = nn.Dropout(0.2)
        self.act2 = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        h = self.act1(self.ln1(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(self.fc2(h))
        return self.act2(h + x)

class improvment_MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.fc_in = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.ln_in = nn.LayerNorm(opt.ndh)
        self.act_in = nn.LeakyReLU(0.2, True)
        self.resblock1 = ResidualBlock(opt.ndh)
        self.resblock2 = ResidualBlock(opt.ndh)
        self.fc_out = nn.Linear(opt.ndh, 1)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.act_in(self.ln_in(self.fc_in(h)))
        h = self.resblock1(h)
        h = self.resblock2(h)
        out = self.fc_out(h)
        return out

############################################
class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class Dis_Embed_Att(nn.Module):
    def __init__(self, opt):
        super(Dis_Embed_Att, self).__init__()
        self.fc1 = nn.Linear(opt.embedSize+opt.attSize, opt.nhF)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.nhF, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, input):
        h = self.lrelu(self.fc1(input))
        h = self.fc2(h)
        return h

