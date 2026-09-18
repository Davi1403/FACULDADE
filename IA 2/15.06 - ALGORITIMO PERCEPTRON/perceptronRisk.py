import pandas as pd
import numpy as np
from random import randrange

def read_file(filename):
    data = pd.read_csv(filename, header=None)
    atributes = data.iloc[:,:-1].astype(float).values.tolist() # [linhas, colunas] (:) todos (:-1) todos exceto o ultimo  
    classes = data.iloc[:,-1].tolist() # [todos,somente o ultimo]
    return atributes, classes

def treino_rna(data, classes, nx, ny, target):

    # VARIAVEIS RNA
    x = np.zeros(nx+1, float)
    time = 100 # QTD. LOOPS DE TREINAMENTO
    alfa = 0.4 # CONSTANTE DE TREINAMENTO
    numSamples = len(data)

    # PASSO 1 - GERAR PESOS ALEATORIAMENTE
    weights = np.random.uniform(-1, 1, nx+1)

    # PASSO 2 - ENQUANTO VERDADE FAÇA (TREINAMENTO)
    for i in range(time):
        # PASSO 3 - SORTEIA UM PAR DE TREINAMENTO
        line = randrange(numSamples)

        # PASSO 4 - ATRIBUIR ATRIBUTOS AOS NEURONIOS(X[I])
        for i in range(nx): # PUXA TODOS OS DADOS DA LINHA PARA O VETOR X
            x[i] = data[line][i]
        x[nx] = 1 # DEFINE BIAS COMO 1
        if classes[line] == target: # SE A CLASSE DA LINHA FOR IGUAL A TARGET (1) SE NÃO (0)
            answer = 1
        else:
            answer = 0

        # PASSO 5 - PROPAGAR SINAL PARA SAIDA(SOMATORIA)
        y = 0
        for i in range(nx+1):
            y+= x[i]*weights[i]
        # PASSO 7 - GERAR A SAIDA DA REDE
        if y>0:
            real_answer = 1
        else:
            real_answer = 0
        if answer != real_answer: # PASSO 8 - ATUALIZAR OS PESOS
            for i in range(nx+1):
                weights[i] += alfa*(answer-real_answer)*x[i]
    return weights


def test(data, classes, nx, ny, target, weights):
    x = np.zeros(nx+1, float)
    numSamples = len(data)
    accuracy = 0

    # PASSO 2 - ENQUANTO VERDADE FAÇA
    for line in range(numSamples):
        for i in range(nx):  # PUXA TODOS OS DADOS DA LINHA PARA O VETOR X
            x[i] = data[line][i]
        x[nx] = 1  # DEFINE BIAS COMO 1
        if classes[line] == target:  # SE A CLASSE DA LINHA FOR IGUAL A TARGET (1) SE NÃO (0)
            answer = 1
        else:
            answer = 0

        y = 0
        for i in range(nx + 1):
            y += x[i] * weights[i]
        # PASSO 7 - GERAR A SAIDA DA REDE
        if y > 0:
            real_answer = 1
        else:
            real_answer = 0
        if answer == real_answer:
            accuracy += 1
    return accuracy*100/numSamples




atributes, classes = read_file("data.txt")


NX = len(atributes[0]) # QUANTIDADE DE ATRIBUTOS
NY = 1 # QUANTIDADE DE SAIDAS

#target = "high_risk"
#target = "mid_risk"
target = "low_risk"

weights = treino_rna(atributes, classes, NX, NY, target)
print(weights)
accuracy = test(atributes, classes, NX, NY, target, weights)
print("accuracy = ", round(accuracy,2))