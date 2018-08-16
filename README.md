# SistemasAutonomos-DL

O objetivo deste trabalho prático é o desenvolvimento de um modelo de Deep Learning para previsão de vendas. Podemos entender este problema como sendo de regressão, onde queremos prever as vendas do mês seguinte com dados de meses anteriores. O dataset fornecido contém o registo de vendas e custos de publicidade de um produto dietético durante o período de tempo de 36 meses (3 anos). Foi também definido que os dois primeiros anos seriam utilizados para treino, sendo o último usado para testes.
	Sendo o dataset uma série temporal, onde a ordem das instâncias importa, foi implementado então um modelo baseado em Long-Short Term Machines (LSTM), com  recurso ao Keras e Tensorflow. 
