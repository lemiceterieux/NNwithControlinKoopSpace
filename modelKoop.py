import torch
from torch import nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.cuda.FloatTensor')

class DeorEncoder(nn.Module):
    def __init__(self, inp, out, dropout=.1, layers=2, neurons=1000):
        super(DeorEncoder, self).__init__()

        L = [nn.Linear(inp, neurons), nn.ReLU()]
        for i in range(layers):
            L += [nn.Linear(neurons,neurons), nn.ReLU()]
        L += [nn.Linear(neurons, out)]
        self.modelMu = nn.Sequential(*L)

    def forward(self, y):
        return self.modelMu(y)

class KoopmanEmbed(nn.Module):

    def __init__(self, nAut=3,nControl=2, num_real=0, num_complex=2, time_steps=10,
            bneurons=400, eneurons=400):
        super(KoopmanEmbed, self).__init__()

        self.num_real = num_real
        self.num_complex = num_complex
        self.time_steps=time_steps
        self.encoder = DeorEncoder(nAut,num_real + 2*num_complex, layers=2, neurons=eneurons)
        self.decoder = DeorEncoder(num_real+2*num_complex,nAut, layers=2, neurons=eneurons)
        self.bShape = num_real + 2*num_complex

        self.B = nn.Sequential(
                nn.Linear((num_real + num_complex),bneurons),
                nn.ReLU(),
                nn.Linear(bneurons,bneurons),
                nn.ReLU(),
                nn.Linear(bneurons,bneurons),
                nn.ReLU(),
                nn.Linear(bneurons, self.bShape*(num_real + 2*num_complex))
                )

        self.EtoB = nn.Sequential(
                nn.Linear(nControl,bneurons),
                nn.ReLU(),
                nn.Linear(bneurons,bneurons),
                nn.ReLU(),
                nn.Linear(bneurons,bneurons),
                nn.ReLU(),
                nn.Linear(bneurons,num_real + 2*num_complex)
            )

        self.eigList = nn.ModuleList([])
        self.BList = nn.ModuleList([])
        for i in range(num_complex):
            eig = nn.Sequential(
                    nn.Linear(1,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,2)
                    )

            self.eigList.append(eig)
            B = nn.Sequential(
                    nn.Linear(1,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,2)
                    )
            self.BList.append(B)

        for i in range(num_real):
            eig = nn.Sequential(
                    nn.Linear(1,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,1)
                    )
            self.eigList.append(eig)
            B = nn.Sequential(
                    nn.Linear(1,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,bneurons),
                    nn.ReLU(),
                    nn.Linear(bneurons,1)
                    )
            self.BList.append(B)
        self.w = nn.Parameter(torch.randn(0))

    def forward(self, inn, trainAut=True):
        if trainAut:
            return self.rForward(inn)
        else:
            return self.predict(inn)

    def odeint(self, X, G):
        ret = [torch.zeros(*X[0].shape[:-1], self.num_real + 2*self.num_complex)]
        for i in range(X.shape[0]):
            ret += [ret[-1] + G*X[i].matmul(self.odeA)]
        return torch.stack(ret)[1:]

    def eigVecs(self, radius):
        vecsList = []
        for i in range(self.num_complex):
            vecsList.append(self.eigList[i](radius[..., [i]]))
        for i in range(self.num_real):
            vecsList.append(self.eigList[i+self.num_complex](radius[...,[i+self.num_complex]]))
        return torch.cat(vecsList,-1)

    def composeB(self, radius):
        vecsList = []
        for i in range(self.num_complex):
            vecsList.append(self.BList[i](radius[..., [i]]))
        for i in range(self.num_real):
            vecsList.append(self.BList[i+self.num_complex](radius[...,[i+self.num_complex]]))
        return torch.cat(vecsList,-1)


    def solveJordanP(self, eig, phi):
        mu = eig[:,:self.num_complex,None,None]
        omega = eig[:,self.num_complex:2*self.num_complex,None]
        Reig = eig[:,2*self.num_complex:]

        phiR = phi[:,2*self.num_complex:]*Reig.exp()
        rCscale = mu.exp()

        iCsin = torch.sin(omega)
        iCcos = torch.cos(omega)
        block = torch.cat((iCcos, -iCsin, iCsin, iCcos),-1).view(*omega.shape[:-1], 2,2)
        eigJ = rCscale*block
        temp = phi[:,:2*self.num_complex].view(*phi.shape[:-1], self.num_complex,2,1)
        phiC = eigJ.matmul(temp).view(*phi.shape[:-1], 2*self.num_complex)
        return torch.cat((phiC, phiR),-1)

    def solveControlP(self, eig, phi):
        mu = eig[...,:self.num_complex,None,None]
        omega = eig[...,self.num_complex:2*self.num_complex,None]
        Reig = eig[...,2*self.num_complex:]

        phiR = phi[...,2*self.num_complex:]*Reig.exp()
        rCscale = mu.exp()

        iCsin = torch.sin(omega)
        iCcos = torch.cos(omega)
        block = torch.cat((iCcos, -iCsin, iCsin, iCcos),-1).view(*omega.shape[:-1], 2,2)
        eigJ = rCscale*block
        temp = phi[...,:2*self.num_complex].view(*phi.shape[:-1], self.num_complex,2,1)
        phiC = eigJ.matmul(temp).view(*phi.shape[:-1], 2*self.num_complex)
        return torch.cat((phiC, phiR),-1)

    def solveJordanR(self, eig, phi):
        mu = eig[:,:,:self.num_complex,None,None]
        omega = eig[:,:,self.num_complex:2*self.num_complex,None]
        Reig = eig[:,:,2*self.num_complex:]

        phiR = phi[:,:,2*self.num_complex:]*Reig.exp()
        rCscale = mu.exp()

        iCsin = torch.sin(omega)
        iCcos = torch.cos(omega)
        block = torch.cat((iCcos, -iCsin, iCsin, iCcos),-1).view(*omega.shape[:-1], 2,2)
        eigJ = rCscale*block
        temp = phi[:,:,:2*self.num_complex].view(*phi.shape[:-1], self.num_complex,2,1)
        phiC = eigJ.matmul(temp).view(*phi.shape[:-1], 2*self.num_complex)
        return torch.cat((phiC, phiR),-1)


    def rForward(self, inn):
        # Encoding
        shinn = inn.shape
        inn = inn.view(inn.shape[0], -1, inn.shape[-1])


        phi = self.encoder(inn[:-1])# + .001*torch.randn(*inn[:-1].shape))

        # Eigen Values

        radius = [((phi[:-self.time_steps,:,i:i+2]**2).sum(-1, keepdim=True)) for i in range(0,self.num_complex*2, 2)]
        radius = (torch.cat(radius + [phi[:-self.time_steps,:,2*self.num_complex:]**2],-1))

        eig = self.eigVecs(radius)

        phie1 = [phi[:-self.time_steps]]        

        for i in range(1,self.time_steps+1):

            # phi1 from Koopman
            phie1.append(self.solveJordanR(eig,phie1[-1]))

            radius = [((phie1[-1][:,:,j:j+2]**2).sum(-1, keepdim=True)) for j in range(0,self.num_complex*2, 2)]
            radius = torch.cat(radius + [phie1[-1][:,:,2*self.num_complex:]**2],-1)

            eig = self.eigVecs(radius)

        phie1 = torch.stack(phie1)
        phi = phi.view(phi.shape[0], -1, *shinn[2:-1], phi.shape[-1])
        phie1 = phie1.view(phie1.shape[0], phie1.shape[1], -1, *shinn[2:-1], phi.shape[-1])

        # Decode
        decode1 = self.decoder(phie1)

        return decode1, phie1, phi

    def predict(self, inn,time=None):
        shinn = inn.shape

        inn = inn.view(inn.shape[0],-1,inn.shape[-1])
        if time is None:
            time = inn.shape[0]
        with torch.no_grad():
            phi = self.encoder(inn[0])

        tempi = self.EtoB(inn[:,:,:-2])

        radius = [((phi[:,i:i+2]**2).sum(-1, keepdim=True)) for i in range(0,self.num_complex*2, 2)]
        radius = (torch.cat(radius + [phi[:,2*self.num_complex:]**2],-1))
        eig = self.eigVecs(radius)
        eigbar = eig
        phie1 = [phi]

        B = self.composeB(radius)
        uB = self.solveControlP(B, phi[-1]- tempi[0])

        phibar = [phie1[-1]]
        err = tempi[0] - phie1[-1]

        for i in range(1,time):
            # phibar

            phibar += [self.solveJordanP(eigbar, phibar[-1])]

            # phi1 from Koopman + control
            phie1.append(self.solveJordanP(eig, phie1[-1]))
            phie1[-1] = phie1[-1] + uB

            radius = [((phie1[-1][:,j:j+2]**2).sum(-1, keepdim=True)) for j in range(0,self.num_complex*2, 2)]
            radius = (torch.cat(radius + [phie1[-1][:,2*self.num_complex:]**2],-1))
    
            radiusb = [((phibar[-1][:,j:j+2]**2).sum(-1, keepdim=True)) for j in range(0,self.num_complex*2, 2)]
            radiusb = (torch.cat(radiusb + [phibar[-1][:,2*self.num_complex:]**2],-1))
    
            eig = self.eigVecs(radius)
            eigbar = self.eigVecs(radiusb)

           
            B = self.composeB(radius)
            uB = self.solveControlP(B, phi[-1]- tempi[i])


        phie1 = torch.stack(phie1)
        phibar = torch.stack(phibar)
        phibar = phibar.view(phibar.shape[0], -1, *shinn[2:-1], phi.shape[-1])
        phie1 = phie1.view(phie1.shape[0],  -1, *shinn[2:-1], phi.shape[-1])

        # Decode
        with torch.no_grad():
            decode1 = self.decoder(phie1)
            decodebar = self.decoder(phibar)


        tempi = tempi.view(tempi.shape[0], -1, *shinn[2:-1], tempi.shape[-1])
        return decode1, phie1, tempi[:-1], phibar, decodebar

def crit(inp, targ, time, it):
    decod = inp[0][0]
    phie1 = inp[0][1]

    phi1 = inp[0][2]

    traject = inp[1][0]
    trajphi = inp[1][1]
    trajphibar = inp[1][3]
    decodebar = inp[1][4]
    decodtemp = inp[1][2]


    B = inp[2].B

    Lrecon = 0
    Lrecon = Lrecon + (decod[0] - (targ[:-(phie1.shape[0])]))**2
    Lrecontraj = 0
    Lrecontraj = Lrecontraj + (traject - (targ[:time]))**2
    Ptraj = (traject - traject.mean(0))*(targ[:time] - targ[:time].mean(0))
    Ptraj = Ptraj/(traject.std(0)*targ[:time].std(0))

    Lkooptraj = (phi1[:time] - trajphi)**2
    mTup = [i for i in range(1,len(Lrecontraj.shape))]
    Lrecontraj = Lrecontraj.mean(mTup).sum()

    Lkooptraj = Lkooptraj.mean(mTup).sum()

    l2B = torch.tensor([0])
    for param in inp[2].eigList:
        for p in param.parameters():
            l2B = l2B + torch.norm(p)
    for param in inp[2].BList:
        for p in param.parameters():
            l2B = l2B + torch.norm(p)

    Lkoop = [] 
    Lfut = [] 

    Pearson = []
    for i in range(1,phie1.shape[0]-1):
        Lkoop += [(phie1[i] - phi1[i:-(phie1.shape[0]-1)+i])**2]
        temp = 0
        temp = temp + (decod[i] - targ[i:-(phie1.shape[0])+i])**2
        Lfut += [temp]
        Corr = (decod[i] - decod[i].mean(0))*(targ[i:-(phie1.shape[0])+i] - targ[i:-(phie1.shape[0])+i].mean(0))
        Corr = Corr/(decod[i].std(0)*targ[i:-(phie1.shape[0])+i].std(0))
        Pearson += [Corr.mean(0)]
    Pearson = torch.stack([Pearson[-1]] + [Ptraj.mean(0)])
    Pearsonstd = torch.std(Pearson)
    Pearsonmean = torch.mean(Pearson[...,-2:])
    Lfut = torch.stack(Lfut)
    Lkoop = torch.stack(Lkoop)
    Linf = torch.max(Lrecon + Lfut)
    Lrecon = Lrecon.mean(mTup).sum()

    mTup = [i for i in range(1, len(Lfut.shape))]
    Lfut = Lfut.mean(mTup).sum()
    mTup = [0] + [i for i in range(2, len(Lkoop.shape))]
    Lkoop = Lkooptraj

    lossreg = .1*(Lrecon + Lfut) + Lkoop + 1e-9*Linf
   
    return lossreg, Lrecon, Lfut, Lkoop, Pearsonmean, Pearsonstd, Pearson
