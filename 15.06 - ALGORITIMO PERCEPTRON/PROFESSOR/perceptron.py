import pandas as pd
import numpy as np
from random import randrange

def ler_Arquivo(arquivo): 
    df = pd.read_csv(arquivo, header=None) 
    atributos = df.iloc[:, :-1].astype(float).values.tolist() 
    classes = df.iloc[:, -1].tolist() 
    return atributos, classes

def rna_treino1(d,cl,nx,ny,ta):
    # Variáveis da RNA
    x  = np.zeros(nx+1,float)
    tempo = 1000
    alfa = 0.4
    ns = len(d)
    v  = np.random.uniform(-1,1,nx+1)
    
    for t in range(tempo):
        linha = randrange(ns)
        for i in range(nx):
            x[i] = d[linha][i]
        x[nx] = 1
        if cl[linha]==ta:
            se = 1
        else:
            se = 0
        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if y>0:
            sr = 1
        else:
            sr = 0
        if se!=sr:
            for i in range(nx+1):
                v[i] += alfa*(se-sr)*x[i]
    return v

def rna_teste1(d,cl,nx,ny,ta,v):
    x  = np.zeros(nx+1,float)
    ns = len(d)
    ac=0
    for linha in range(ns):
        for i in range(nx):
            x[i] = d[linha][i]
        x[nx] = 1
        if cl[linha]==ta:
            se = 1
        else:
            se = 0
        y = 0
        for i in range(nx+1):
            y += x[i]*v[i]
        if y>0:
            sr = 1
        else:
            sr = 0
        if se==sr:
            ac += 1

    return ac*100/ns

#------------------------------------------------------
#                 MÓDULO PRINCIPAL
#------------------------------------------------------
arquivo = "iris.txt"
dados, classe = ler_Arquivo(arquivo)
NX = len(dados[0])
NY = 1

target = "Iris-setosa"
v = rna_treino1(dados,classe,NX,NY,target)
print(v)
ac = rna_teste1(dados,classe,NX,NY,target,v)
print("Acurácia: ",round(ac,2))

