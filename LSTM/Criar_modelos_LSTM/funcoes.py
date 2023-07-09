import os
import time

import keras
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy
import ta as ta
import yfinance as yf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import GRU, LSTM, RNN, Dense, Dropout
from keras.models import Sequential, load_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features
from ta.utils import dropna
from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk


def ReduzirRuido(base,pulo=5):
    
    de = 0
    ate = pulo
    
    df = base.tail(len(base)-(len(base)%pulo))

    df = df.reset_index()

    lista_diario = list()
    
    for i in range(int(len(df)/pulo)):

        lista_diario.append(df[de:ate])

        de += pulo
        ate += pulo
    
    
    abertura = list()
    fechamento = list()
    maximo = list()
    minimo = list()
    datas = list()

    i = 0
    for triario in lista_diario:

        datas.append(triario["Date"][i])
        abertura.append(triario["Open"][i])
        fechamento.append(triario["Close"][i])
        maximo.append(triario["High"].max())
        minimo.append(triario["Low"].min())

        i+=pulo
        
        
    triario = zip(datas,abertura,fechamento,minimo,maximo)
    
    triario_df = pd.DataFrame(triario,columns=["Date","Open","Close","Low","High"])
    
    triario_df = triario_df.set_index("Date")

    return triario_df




def preparar_dados_financeiros(ticker,pulo,remove_out=False):

    print(ticker)
    base = yf.download(ticker)

    if base["Open"].tail(1)[0] == 0:
        base = base[:len(base)-1]

    if pulo != 0 and pulo != 1:
        
        base = ReduzirRuido(base,pulo)

    if remove_out:
        return remover_outliers(base)
    else:
        return base


def remover_outliers(base):

    z_scores = scipy.stats.zscore(base)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    new_base = base[filtered_entries]

    return new_base


def GetCci(base, normalizar=1):

    cci_config = [20, 0.015]

    resultados_cci = ta.trend.CCIIndicator(
        base["High"], base["Low"], base["Close"], cci_config[0], cci_config[1], False)

    cci_df = pd.DataFrame(resultados_cci.cci())

    cci_df.dropna(inplace=True)

    if normalizar == 1:

        cci_normalizado = Normalizar(cci_df)

        return cci_normalizado

    return cci_df


def GetTsi(base, gaussian_knots, gaussian_sigma, ewm_span=20):

    tsi_config = [25, 13]

    resultados_tsi = ta.momentum.TSIIndicator(
        base["Close"], tsi_config[0], tsi_config[1], False)

    tsi_df = pd.DataFrame(resultados_tsi.tsi())

    tsi_df.dropna(inplace=True)

    # Suavizando TSI com médias móveis exponenciais
    tsi_df["ewm"] = tsi_df['tsi'].ewm(span=ewm_span).mean()*1.2
    # ------------------------------------------

    # Suavizanto TSI com gaussian smoother
    tsi_np = tsi_df["tsi"].to_numpy()
    tsi_np.reshape(1, len(tsi_np))

    smoother = GaussianSmoother(n_knots=gaussian_knots, sigma=gaussian_sigma)
    smoother.smooth(tsi_np)

    tsi_df["gaussian"] = smoother.smooth_data[0]
    # ------------------------------------------

    return tsi_df


def GetRsi(base, normalizar=1):

    rsi_config = [14, 3, 3]

    resultados_rsi = ta.momentum.RSIIndicator(
        base["Close"], rsi_config[0], False)

    rsi_df = pd.DataFrame(resultados_rsi.rsi())

    rsi_df.dropna(inplace=True)

    if normalizar == 1:

        rsi_normalizado = Normalizar(rsi_df, 0)

        return rsi_normalizado

    return rsi_df


def Normalizar(Oscilador, coluna):

    normalizador = MinMaxScaler(feature_range=(0, 1))

    if coluna == "tsi":
        Oscilador_treinamento = Oscilador.iloc[:, 0:1].values

    if coluna == "ewm":
        Oscilador_treinamento = Oscilador.iloc[:, 1:2].values

    if coluna == "gaussian":
        Oscilador_treinamento = Oscilador.iloc[:, 2:3].values

    Oscilador_normalizado = normalizador.fit_transform(Oscilador_treinamento)

    return Oscilador_normalizado


def preparar_dados_para_treinamento(anteriores, base_treinamento_normalizada):

    previsores = []
    preco_real = []

    for i in range(anteriores, len(base_treinamento_normalizada)):

        previsores.append(base_treinamento_normalizada[i-anteriores:i, 0])
        preco_real.append(base_treinamento_normalizada[i, 0])

    previsores, preco_real = np.array(previsores), np.array(preco_real)
    previsores = np.reshape(
        previsores, (previsores.shape[0], previsores.shape[1], 1))

    return previsores, preco_real


def criarRedeNeural(previsores, preco_real, filepath, epocas=300, validacao_cruzada=0, ativacao="linear", otimizador="adam", minimo_delta=1e-15, paciencia_es=10, batch=40):

    regressor = Sequential()

    # 1º
    regressor.add(LSTM(units=70, return_sequences=True,
                  input_shape=(previsores.shape[1], 1)))
    regressor.add(Dropout(0.3))

    # 2º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 3º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 4º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))

    # 5º
    regressor.add(LSTM(units=70, return_sequences=True))
    regressor.add(Dropout(0.3))
    '''
    #6º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #7º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #8º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #9º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #10º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #11º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.2))
    
    #12º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.2))
    
    #13º
    regressor.add(LSTM(units=60,return_sequences=True))
    regressor.add(Dropout(0.2))
    
    #14º
    regressor.add(LSTM(units=80,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #15º
    regressor.add(LSTM(units=100,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #16º
    regressor.add(LSTM(units=100,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #17º
    regressor.add(LSTM(units=100,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #18º
    regressor.add(LSTM(units=100,return_sequences=True))
    regressor.add(Dropout(0.3))
    
    #19º
    regressor.add(LSTM(units=100,return_sequences=True))
    regressor.add(Dropout(0.3))
    '''
    # 20º
    regressor.add(LSTM(units=70))
    regressor.add(Dropout(0.3))

    # 21º
    regressor.add(Dense(units=1, activation=ativacao))

    regressor.compile(optimizer=otimizador, loss='mean_squared_error', metrics=[
                      'mean_absolute_error'])

    es = EarlyStopping(monitor="loss", min_delta=minimo_delta,
                       patience=paciencia_es, verbose=1)
    rlr = ReduceLROnPlateau(monitor="loss", factor=0.06, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath=filepath, monitor="loss",
                          save_best_only=True, verbose=1)

    if validacao_cruzada == 1:

        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(previsores):
            X_train, X_test = previsores[train_index], previsores[test_index]
            y_train, y_test = preco_real[train_index], preco_real[test_index]

            regressor.fit(X_train, y_train, epochs=epocas,
                          batch_size=batch, callbacks=[es, mcp])
            score = regressor.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else:

        regressor.fit(previsores, preco_real, epochs=epocas,
                      batch_size=batch, callbacks=[es, mcp])

    return regressor


def criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout, epocas=300, validacao_cruzada=0, loss_='mean_squared_error', ativacao="linear", otimizador="adam", minimo_delta=1e-15, paciencia_es=10, batch=40):

    qtd_camadas -= 2

    regressor = Sequential()

    i = 0
    while i < qtd_camadas:

        if i == 0:
            regressor.add(LSTM(units=qtd_neuronios, return_sequences=True,
                          input_shape=(previsores.shape[1], 1)))
            regressor.add(Dropout(dropout))

        else:
            regressor.add(LSTM(units=qtd_neuronios, return_sequences=True))
            regressor.add(Dropout(dropout))

        i += 1

    regressor.add(LSTM(units=qtd_neuronios))
    regressor.add(Dropout(dropout))

    regressor.add(Dense(units=1, activation=ativacao))

    regressor.compile(optimizer=otimizador, loss=loss_,
                      metrics=['mean_absolute_error'])

    es = EarlyStopping(monitor="loss", min_delta=minimo_delta,
                       patience=paciencia_es, verbose=0)

    rlr = ReduceLROnPlateau(monitor="loss", factor=0.06, patience=5, verbose=0)
    
    mcp = ModelCheckpoint(filepath=filepath, monitor="loss",
                          save_best_only=True, verbose=0)

    if validacao_cruzada == 1:

        kf = KFold(n_splits=5, shuffle=True)

        for train_index, test_index in kf.split(previsores):
            X_train, X_test = previsores[train_index], previsores[test_index]
            y_train, y_test = preco_real[train_index], preco_real[test_index]

            regressor.fit(X_train, y_train, epochs=epocas,
                          batch_size=batch, callbacks=[es, mcp], verbose=0)
            score = regressor.evaluate(X_test, y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    else:

        regressor.fit(previsores, preco_real, epochs=epocas,
                      batch_size=batch, callbacks=[es, mcp], verbose=0)

    return regressor



def Gaussian_1(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 40

    base = base.tail(1257)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 100
    qtd_camadas = 8
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 32

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 1 terminado ", filepath)

    return 1



def Gaussian_2(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 15

    base = base.tail(1257)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 60
    qtd_camadas = 6
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 32

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 2 terminado ", filepath)

    return 1



def Gaussian_3(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 40

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 60
    qtd_camadas = 10
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 40

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 3 terminado ", filepath)

    return 1



def Gaussian_4(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 15

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 60
    qtd_camadas = 7
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 40

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 4 terminado ", filepath)

    return 1



def Gaussian_5(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 40

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 70
    qtd_camadas = 10
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 40

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 5 terminado ", filepath)

    return 1



def Gaussian_6(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 90

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 100
    qtd_camadas = 10
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 32

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 6 terminado ", filepath)

    return 1



def Gaussian_10(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 15

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 80
    qtd_camadas = 6
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-15
    paciencia_es = 10
    batch = 20

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 10 terminado ", filepath)

    return 1



def Gaussian_11(base, filepath, k_nots=80, sigma=0.001):

    anteriores = 15

    base = base.tail(1760)

    tsi = GetTsi(base, k_nots, sigma)

    normalizado = Normalizar(tsi, "gaussian")

    previsores, preco_real = preparar_dados_para_treinamento(
        anteriores, normalizado)

    # regressor = criarRedeNeural(previsores,preco_real,"Modelos_SOMA3\TSI_Gaussian_4.h5",epocas=200)

    qtd_neuronios = 60
    qtd_camadas = 12
    dropout = 0.3
    epocas = 200
    validacao_cruzada = 0
    loss_ = 'mean_squared_error'
    ativacao = 'linear'
    otimizador = 'adam'
    minimo_delta = 1e-10
    paciencia_es = 10
    batch = 20

    modelo = criarRedeNeural_custom(previsores, preco_real, filepath, qtd_neuronios, qtd_camadas, dropout,
                                    epocas, validacao_cruzada, loss_, ativacao, otimizador, minimo_delta, paciencia_es, batch)

    print("Gaussian 11 terminado ", filepath)

    return 1


def Criar_modelos_gaussian(tickers,pulo=0,remove_out=False):

    if pulo != 0 and pulo != 1:
                
        pasta = str(pulo)+"_em_"+str(pulo)+"/"
        
    else:   
        
        pasta = "Diario/"
        

    

    rfs = []
    retornos = []
    i = 0

    for ticker in tickers:

        ticker_ = ticker[0].split(".")[0]

        Modelos_LSTM = "Modelos_LSTM"

        if os.path.exists(Modelos_LSTM) == False:
            os.mkdir(Modelos_LSTM)


        Modelos_ativo = Modelos_LSTM+"/"+ticker_

        if os.path.exists(Modelos_ativo) == False:
            os.mkdir(Modelos_ativo)        

        filepath = Modelos_ativo +"/"+pasta
        
        if os.path.exists(filepath) == False:
            os.mkdir(filepath)        

        base = preparar_dados_financeiros(ticker[0],pulo)

        g_k_nots = ticker[1]
        g_sigma = ticker[2]

        print("Começando treinamento do ativo "+ticker_)

        Gaussian_1(base, filepath +"_Gaussian_1.h5", g_k_nots, g_sigma)
        Gaussian_2(base, filepath +"_Gaussian_2.h5", g_k_nots, g_sigma)
        Gaussian_3(base, filepath +"_Gaussian_3.h5", g_k_nots, g_sigma)
        Gaussian_4(base, filepath +"_Gaussian_4.h5", g_k_nots, g_sigma)
        Gaussian_5(base, filepath +"_Gaussian_5.h5", g_k_nots, g_sigma)
        Gaussian_6(base, filepath +"_Gaussian_6.h5", g_k_nots, g_sigma)
        Gaussian_10(base, filepath +"_Gaussian_10.h5", g_k_nots, g_sigma)
        Gaussian_11(base, filepath +"_Gaussian_11.h5", g_k_nots, g_sigma)
        

        i += 1
