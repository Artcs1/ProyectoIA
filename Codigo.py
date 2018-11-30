import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random
import itertools

def PCA(data,comp = 2):
    n = data.shape[1]
    # matriz de covariância
    cov_matrix = np.cov(data.astype(float), rowvar=False)
   
    # autovetores e autovalores
    
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_matrix)
    
    eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

    # ordenar autovetores e autovalores 
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    ind = 1
    suma = 0 
    print("Autovalores em forma creciente")
    print("------------------------------------------")
    for i in eig_pairs:
        print('| PCA'+ str(ind) + '\t | \t ' + str(i[0]) + '\t |')
        ind = ind+1
        suma = suma + i[0];
    print("------------------------------------------")    
    
    ind = 1
    suma_porcentaje = 0
    print("\nResponsabilidade na variância")
    print("--------------------------")
    for i in eig_pairs:
        print('| PCA'+ str(ind) + '\t | \t ' + str(round((i[0]/suma) * 100,2)) + '%\t |')
        if ind - 1 < comp :
            suma_porcentaje = suma_porcentaje + i[0]
        ind = ind+1
    print("--------------------------")
    print("\nRedução da dimensionalidade com "+ str(comp) + " componentes,é responsavel do " + str(suma_porcentaje/suma * 100) + "% da variância")
    
    
    matrix_w = np.hstack((eig_pairs[0][1].reshape(n,1), eig_pairs[1][1].reshape(n,1)))
    
    transformed = matrix_w.T.dot(data.T)
    return transformed


def graficar(transformed,Y):
    
    ids1 = np.where(Y == 1)
    ids2 = np.where(Y == 2)
    ids3 = np.where(Y == 3)
    
    plt.plot(transformed[0,ids1[0]], transformed[1,ids1[0]], 'o', markersize=7, color='blue', alpha=0.5, label='Batata - Saudavel')
    plt.plot(transformed[0,ids2[0]], transformed[1,ids2[0]], '^', markersize=7, color='red', alpha=0.5, label='Batata - praga - leve')
    plt.plot(transformed[0,ids3[0]], transformed[1,ids3[0]], 'X', markersize=7, color='green', alpha=0.5, label='Bata - praga - tardia')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()
    plt.title('grafico  da dimensionalidade com 2 componentes')

    plt.show()
    return 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues): # matriz de confusão

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')

def sigmoid(net): # função sigmoide
    return (1/(1+np.exp(-net)))

def derivadanet(net):
    return (net*(1-net))

def feedforward(X,model,function = sigmoid):
    
    fnetH = []
    Whidden = model[3]
    Woutput = model[4]
    
    X = np.concatenate((X,np.ones(1)),axis = 0)
    
    for i in range (len(model[1])):
        netH = np.dot(Whidden[i],X)
        fnetH.append(function(netH))
        X = np.concatenate((fnetH[i],np.ones(1)),axis = 0)
        
    netO = np.dot(Woutput,X)
    fnetO = function(netO)
    
    
    return [fnetO,fnetH]

def backpropagation(model,X,Y,eta = 0.1,momentum = 0.2 ,threshold = 1e-7, dnet = derivadanet,maxiter = 80):
    sError = 2*threshold
    c = 0
    while sError > threshold and c < maxiter: # criterio de parada
        sError = 0
        
        tam = Y.shape[0]
        
        
        for i in np.arange(tam):
        
            Xi = X[i,]
            Yi = Y[i,]
        
            results = feedforward(Xi,model);

            O = results[0]
            #Error
            Error = Yi-O
        
            sError = sError + np.sum(np.power(Error,2))
        
            #Treinamento capa de salida
            h = len(model[1])
            
            Hp = np.concatenate((results[1][h-1],np.ones(1)),axis=0)
            Hp = np.reshape(Hp,(model[1][h-1]+1,1))
            
            deltaO = Error * dnet(results[0])
            dE2_dw_O = np.dot((-2*deltaO.reshape((deltaO.shape[0],1))),np.transpose(Hp)) 
            Voutput = np.zeros(dE2_dw_O.shape)
            Voutput = momentum*Voutput + dE2_dw_O
            #Treinamento capa intermedia
        
            deltaH = []
            dE2_dw_h = []
            Vhidden = []
            
            delta = deltaO
            Wk = model[4][:,0:model[1][h-1]]
        
            for i in range(h):
                deltaH.append(0)
                dE2_dw_h.append(0)
                Vhidden.append(0)
            
            for h in range(len(model[1])-1,0,-1):
                Xp = np.concatenate((results[1][h-1],np.ones(1)),axis = 0 )
                Xp = np.reshape(Xp,(1,model[1][h-1]+1))
                
                deltaH[h] = np.dot(delta,Wk) 
                dE2_dw_h[h] = deltaH[h].reshape((deltaH[h].shape[0],1)) * (np.dot(-2*dnet(results[1][h]).reshape((results[1][h].shape[0],1)),Xp))
                Vhidden[h] = momentum*Vhidden[h] + dE2_dw_h[h]
                
                
                delta = deltaH[h]
                Wk = model[3][h][:,0:model[1][h-1]]
            
            Xp = np.concatenate((Xi,np.ones(1)),axis=0)
            Xp = np.reshape(Xp,(1,model[0]+1))
            
            deltaH[0] = (np.dot(delta,Wk))
            dE2_dw_h[0] = deltaH[0].reshape((deltaH[0].shape[0],1)) * (np.dot(-2*dnet(results[1][0]).reshape((results[1][0].shape[0],1)),Xp))
            Vhidden[0] =momentum*Vhidden[0] + dE2_dw_h[0]
            #atualização dos pesos
        
            model[4] =  model[4] -  eta*Voutput            
            for i in range(len(model[1])):
                model[3][i] =  model[3][i] -  eta*Vhidden[i] 
        
        #contador
        
        sError = sError / tam
        c = c+1
        #print("iteração ",c)
        #print("Error:",sError)
        #print("\n");
    

    return model
def mlp(Isize = 10,Hsize = [2,4] ,Osize = 3):
    
    # Isize tamano da camada de entrada
    # Osize tamano da camada de salida
    # Hsize tamano de camada oculta

    Whidden = []
    Vmomentum = []
    previous_length = Isize    
    for i in range (len(Hsize)):
        Whidden.append(np.random.random_sample((Hsize[i],previous_length +1)) - 0.5 )
        Vmomentum.append(np.zeros((Hsize[i],previous_length +1)));
        previous_length = Hsize[i]    

    Woutput = np.random.random_sample((Osize,previous_length +1)) - 0.5     
    model = [Isize,Hsize,Osize,Whidden,Woutput,Vmomentum]
    
    return model

def binarizar(Y,siz = 3):
    Y2 = np.zeros((Y.shape[0],siz))
    for i in np.arange(Y.shape[0]):
        Y2[i,int(Y[i])-1] = 1
        
    return Y2

def clasificacion(model,X,Y):
    acierto = 0;
    P = [] 
    tam = Y.shape[0]
    for i in np.arange(tam):
        Yesperado = feedforward(X[i,],model)[0]
        Ya = np.argmax(Yesperado) + 1
        Yi = np.round(Yesperado)
        if np.sum(Yi - Y[i,]) == 0: 
             acierto = acierto +1
        P.append(Ya)
    return [(acierto*100)/tam,P]

def resize(data,siz = 152): # Resize
    data = data[1:]
    setA = np.arange(152)
    idx = random.sample(list(setA),siz) # Amostragem da Data
    dataset = data[idx,:]
    return dataset

def preprocessing(data):
    # media 0 e variância 1
    scale = np.nanstd(data, axis = 0)
    X = data - sp.mean(data,axis = 0)
    return X/scale

def HoldOut(dataset, train_size = 0.7): # Hold Out
    row = dataset.shape[0]
    setA = np.arange(row)
    idx = random.sample(list(setA), int(np.floor(row * train_size ))) # amostragem de dataset 
    setB = idx
    idx2 = np.setdiff1d(setA,setB) # ids do complemento
    traindata = dataset[idx,:]
    testdata = dataset[idx2,:]
    return [traindata,testdata]

def confusion_matrix(output , pred , classe = 3):
    C = np.zeros( (classe , classe ))
    for i in range(len(output)):
        C[output[i]-1][pred[i]-1] = C[output[i]-1][pred[i]-1] + 1
    return C 
    

def metrics(pred,output):
    
    MP = 0
    MR = 0
    Mf1 = 0
    
    for i in np.arange(3): # Calculando precisão , recall ,f1 - medida por cada clase
        c= i+1;
        print('Clase',c)
        PP = np.sum(np.logical_and(output == c, pred == c)) # Verdadero Positivo
        NP = np.sum(np.logical_and(output == c, pred != c)) # Falso Negativo
        PN = np.sum(np.logical_and(output != c, pred == c)) # Verdadero Negativo
        precision = PP/(PP+NP)
        recall = PP/(PP+PN)
        f1 = 2*(precision*recall)/(precision+recall)
        MP = MP + precision
        MR = MR + recall
        Mf1 = Mf1 + f1
        print('---------------------------------')
        print('| Precision\t|\t' + str(round(precision,2)) + '\t|')
        print('| Recall   \t|\t' + str(round(recall,2)) + '\t|')
        print('| f1       \t|\t' + str(round(f1,2)) + '\t|')
        print('---------------------------------\n')
    
    # Promedio de M , MR , Mf1
    print('\n')
    print('--------------------------------------------------')
    print('| Promedio de Precision\t|\t'+ str(MP/3) +'|')
    print('| Promedio de Recall   \t|\t'+str(MR/3) + '|')
    print('| Promedio de f1       \t|\t'+ str(Mf1/3)+'|')
    print('--------------------------------------------------\n')    
    
    cnf_matrix = confusion_matrix(output, pred) # matriz de confusão
    plt.figure()
    class_names = ["Batata Saudavel","Batata plaga leve","Batata con plaga Tardia"]
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Matriz de confusão normalizada')



def batata_test(Data,siz = 0.5,maxiter = 10, eta = 0.1,momentum = 0.0):
 
    DataH = HoldOut(Data) # HoldOut 
    Train = DataH[0] 
    Test = DataH[1]
    
    X = Train[:,0:9] # X treinamento
    X = preprocessing(X) # Normalização
    
    Y = Train[:,9].astype(int) # Y treinamento
    transformed = PCA(X, comp = 2)
    graficar(transformed,Y)
    Y = binarizar(Y)
    
    features = Test[:,0:9] # X Test
    X2 = preprocessing(features) # Normalização
    output = Test[:,9].astype(int) # Y tests
    Y2 = binarizar(output)
    
    M = mlp(Isize = 9, Hsize = [5], Osize = 3)
    trained = backpropagation(M,X,Y,eta = eta , momentum = momentum , maxiter = maxiter)
    
    A = clasificacion(trained,X,Y)
    B = clasificacion(trained,X2,Y2)
    
    print("\nParticionamiento: "+ str(siz) +" Max.Iter: " + str(maxiter) + " Eta: " + str(eta) +  " Momentum:" + str(momentum))    
    print("Error na data de treinamento",A[0])
    print("Error na data de test",B[0])
    print("\n")
    
    output = []
    tam = Y2.shape[0]
    for i in np.arange(tam):
        output.append(np.argmax(Y2[i,]) + 1)
        
    metrics( B[1], output ) 
    
    return [A,B]

def main():
    
    D1 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/7tociclo/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataH.dat') # Ruta dos descriptores da batata saudavel 
    D2 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/7tociclo/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataE.dat') # Ruta dos descriptores da batata com plaga leve 
    D3 = np.loadtxt('C:/Users/nineil/Desktop/Jeffri/USP/7tociclo/Procesamiento de Imagenes/Proyecto_PI/DataDescriptor/BatataT.dat') # Ruta dos descriptores da batata com plaga tardia 
    
    D1 = resize(D1) 
    D2 = resize(D2) 
    D3 = resize(D3) 
    
    Data = np.concatenate((D1,D2,D3),axis = 0 ) # Concatenando as matrices
    
    batata_test(Data)

if __name__ == "__main__":
    main()