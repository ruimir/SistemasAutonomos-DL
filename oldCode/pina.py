from idlelib import history

import numpy as np
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import matplotlib.pyplot as plt
# fixar random seed para se puder reproduzir os resultados
seed = 9
np.random.seed(seed)

# Etapa 1 - preparar o dataset
def read_cvs_dataset(ficheiro, col_label):
    # ler ficheiro csv para matriz numpy, e separar o label que está em col_label (deve ser a ultima coluna)
    dataset = np.loadtxt(ficheiro, delimiter=",")
    print('Formato do dataset: ', dataset.shape)
    input_attributes = dataset[:, 0:col_label]
    output_attributes = dataset[:, col_label]
    print('Formato das variáveis de entrada (input variables): ',input_attributes.shape)
    print('Formato da classe de saída (output variables): ', output_attributes.shape)  # print(X[0])
    # print(Y[0])
    return (input_attributes, output_attributes)

# Etapa 2 - Definir a topologia da rede (arquitectura do modelo)

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
    return model

#util para visualizar a topologia da rede num ficheiro em pdf ou png
def print_model(model,fich):
    from keras.utils import plot_model
    plot_model(model, to_file=fich, show_shapes=True, show_layer_names=True)

# Etapa 3 - Compilar o modelo (especificar o modelo de aprendizagem a ser utilizado pela rede)

#loss - funcão a ser utilizada no calculo da diferença entre o pretendido e o obtido vamos utilizar logaritmic loss para classificação binária: 'binary_crossentropy'
#o algoritmo de gradient descent será o “adam” pois é eficiente
#a métrica a ser utilizada no report durante o treino será 'accuracy' pois trata-se de um problema de classificacao


def compile_model(model):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def fit_model(model,input_attributes,output_attributes):
    history = model.fit(input_attributes, output_attributes, validation_split=0.33, epochs=150, batch_size=10, verbose=2)
    return history

def print_history_accuracy(history):
    print(history.history.keys())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def print_history_loss(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def model_evaluate(model,input_attributes,output_attributes):
    print("###########inicio do evaluate###############################\n")
    scores = model.evaluate(input_attributes, output_attributes)
    print("\n metrica: %s: %.2f%%\n" % (model.metrics_names[1], scores[1]*100))

def model_print_predictions(model,input_attributes,output_attributes):
    previsoes = model.predict(input_attributes)
    # arredondar para 0 ou 1 pois pretende-se um output binário
    LP = []
    for prev in previsoes:
        LP.append(round(prev[0]))
    # LP = [round(prev[0]) for prev in previsoes]
    for i in range(len(output_attributes)):
        print(" Class:", output_attributes[i], " previsão:", LP[i])
        if i > 10: break

def ciclo_completo():
    (input_attributes, output_attributes) = read_cvs_dataset("pima-indians-diabetes.csv", 8)
    model = create_model()
    print_model(model, "model_MLP.png")
    compile_model(model)
    history = fit_model(model, input_attributes, output_attributes)
    print_history_accuracy(history)
    print_history_loss(history)
    model_evaluate(model, input_attributes, output_attributes)
    model_print_predictions(model, input_attributes, output_attributes)

def save_model_json(model,fich):
    model_json = model.to_json()
    with open(fich, "w") as json_file:
        json_file.write(model_json)

def save_model_yaml(model,fich):
    model_yaml = model.to_yaml()
    with open(fich, "w") as yaml_file:
        yaml_file.write(model_yaml)

def save_weights_hdf5(model,fich):
    model.save_weights(fich)
    print("Saved model to disk")

def load_model_json(fich):
    json_file = open(fich, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model

def load_model_yaml(fich):
    yaml_file = open(fich, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    return model_from_yaml(loaded_model_yaml)


def load_weights_hdf5(model,fich):
    model.load_weights(fich)
    print("Loaded model from disk")

# exemplos de utilização destes utilitários
def ciclo_ler_dataset_treinar_gravar():
    (input_attributes,output_attributes) = read_cvs_dataset("pima-indians-diabetes.csv",8)
    model = create_model()
    print_model(model,"model2.png")
    compile_model(model)
    history=fit_model(model,input_attributes,output_attributes)
    print_history_accuracy(history)
    print_history_loss(history)
    model_evaluate(model,input_attributes,output_attributes)
    save_model_json(model,"model.json")
    save_weights_hdf5(model,"model.h5")
    return (input_attributes,output_attributes)

def ciclo_ler_modelo_evaluate_usar(input_attributes,output_attributes):
    model= load_model_json("model.json")
    load_weights_hdf5(model,"model.h5")
    compile_model(model)
    model_evaluate(model,input_attributes,output_attributes)
    model_print_predictions(model,input_attributes,output_attributes)

if __name__ == '__main__':
    #opção 1 - ciclo completo
    #ciclo_completo()
    #opção 2 - ler,treinar o dataset e gravar. Depois ler o modelo e pesos e usar
    (input_attributes, output_attributes)=ciclo_ler_dataset_treinar_gravar()
    ciclo_ler_modelo_evaluate_usar(input_attributes, output_attributes)