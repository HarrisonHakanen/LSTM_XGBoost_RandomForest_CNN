import datetime as dt
import os
import time
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ta as ta
import yfinance as yf
from numpy import arange
from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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


def preparar_dados_para_treinamento(anteriores,base_treinamento_normalizada):

    previsores = []
    preco_real = []

    for i in range(anteriores,len(base_treinamento_normalizada)):

        previsores.append(base_treinamento_normalizada[i-anteriores:i,0])
        preco_real.append(base_treinamento_normalizada[i,0])

    previsores,preco_real = np.array(previsores),np.array(preco_real)
    previsores = previsores
    
    return previsores,preco_real

def GetTsi(base,gaussian_knots,gaussian_sigma,ewm_span=20):
    
    tsi_config=[25,13]

    resultados_tsi = ta.momentum.TSIIndicator(base["Close"],tsi_config[0],tsi_config[1],False)

    tsi_df = pd.DataFrame(resultados_tsi.tsi())
    
    tsi_df.dropna(inplace=True)
    
    #Suavizando TSI com médias móveis exponenciais
    tsi_df["ewm"] = tsi_df['tsi'].ewm(span = ewm_span).mean()*1.2
    #------------------------------------------
    
    #Suavizanto TSI com gaussian smoother
    tsi_np = tsi_df["tsi"].to_numpy()
    tsi_np.reshape(1,len(tsi_np))

    smoother = GaussianSmoother(n_knots=gaussian_knots, sigma=gaussian_sigma)
    smoother.smooth(tsi_np)

    tsi_df["gaussian"] = smoother.smooth_data[0]
    #------------------------------------------
    
    return tsi_df

def Normalizar(Oscilador,coluna):
    
    normalizador = MinMaxScaler(feature_range=(0,1))
    
    if coluna == "tsi":
        Oscilador_treinamento = Oscilador.iloc[:,0:1].values
        
    if coluna == "ewm":
        Oscilador_treinamento = Oscilador.iloc[:,1:2].values
        
    if coluna == "gaussian":
        Oscilador_treinamento = Oscilador.iloc[:,2:3].values
        
    Oscilador_normalizado = normalizador.fit_transform(Oscilador_treinamento)
    
    return Oscilador_normalizado


def Criar_modelo_randomForest(base,anteriores_,filepath,knots_=60,sigma_=0.0003,n_estimators_=100,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1):
    
    #Extrai o tsi
    tsi = GetTsi(base,knots_,sigma_)
    #--------------------------------


    #Faz a normalização do gaussian do tsi
    normalizado = Normalizar(tsi,"gaussian")
    #--------------------------------


    X_train, y_train = preparar_dados_para_treinamento(anteriores_,normalizado)

    forest_model = RandomForestRegressor(
        random_state=1,
        n_estimators=n_estimators_,
        max_depth=max_depth_,
        min_samples_split=min_samples_split_,
        min_samples_leaf=min_samples_leaf_)
    
    
    forest_model.fit(X_train, y_train)
    
    #joblib.dump(forest_model, "RandomForest_4.joblib")
    joblib.dump(forest_model, filepath)
    
    print("Modelo ",filepath, "criado")
    
    return forest_model


def Criar_modelo_rf_1(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=200,max_depth_=100,min_samples_split_=4,min_samples_leaf_=5)
    
    
    return forest_model


def Criar_modelo_rf_2(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,90,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=300,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_3(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=200,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_4(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=400,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_5(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,90,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=400,max_depth_=None,min_samples_split_=4,min_samples_leaf_=5)
    
    
    return forest_model


def Criar_modelo_rf_7(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,300,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=100,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_8(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,300,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=500,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_9(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,500,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=300,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_10(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,500,filepath_,knots_=60,sigma_=0.0003,
        n_estimators_=500,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model

#----------------------------------------------------------------------------------------------

def Criar_modelo_rf_11(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=200,max_depth_=100,min_samples_split_=4,min_samples_leaf_=5)
    
    
    return forest_model


def Criar_modelo_rf_12(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,90,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=300,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_13(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=200,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_14(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,180,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=400,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_15(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,90,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=400,max_depth_=None,min_samples_split_=4,min_samples_leaf_=5)
    
    
    return forest_model


def Criar_modelo_rf_16(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,300,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=100,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_17(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,300,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=500,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_18(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,500,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=300,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelo_rf_19(base,filepath_):
    
    forest_model = Criar_modelo_randomForest(base,500,filepath_,knots_=90,sigma_=0.0003,
        n_estimators_=500,max_depth_=None,min_samples_split_=2,min_samples_leaf_=1)
    
    
    return forest_model


def Criar_modelos_random_forest(ticker_lista,pulo=0):
    
    
    
    for ticker in ticker_lista:
        print(ticker)
        
        df = yf.download(ticker)
        
        if df["Open"].tail(1)[0] == 0:
            df = df[:len(df)-1]

        if pulo != 0 and pulo != 1:
            
            pasta = str(pulo)+"_em_"+str(pulo)+"/"
            df_sem_ruido = ReduzirRuido(df,pulo)
            base = df_sem_ruido.tail(1267) 

        else:   
            
            pasta = "Diario/"
            
            base = df.tail(1267)


        
        ticker_split = ticker.split(".")

        modelos_rf = "Modelos_RF/"
        
        if os.path.exists(modelos_rf) == False:
            os.mkdir(modelos_rf)
        
        modelos_acao = modelos_rf+"Modelos_"+ticker_split[0]+"/"
        
        if os.path.exists(modelos_acao) == False:
            os.mkdir(modelos_acao)
        
        caminho = modelos_acao+pasta
        
        
        
        if os.path.exists(caminho) == False:
            os.mkdir(caminho)
    
        
        
        #Modelos com o gaussiano menos sensíveis
        modelo_rf_1 = Criar_modelo_rf_1(base,caminho+ticker_split[0]+"_RF_1.joblib")
        modelo_rf_2 = Criar_modelo_rf_2(base,caminho+ticker_split[0]+"_RF_2.joblib")
        modelo_rf_3 = Criar_modelo_rf_3(base,caminho+ticker_split[0]+"_RF_3.joblib")
        modelo_rf_4 = Criar_modelo_rf_4(base,caminho+ticker_split[0]+"_RF_4.joblib")
        modelo_rf_5 = Criar_modelo_rf_5(base,caminho+ticker_split[0]+"_RF_5.joblib")
        modelo_rf_7 = Criar_modelo_rf_7(base,caminho+ticker_split[0]+"_RF_7.joblib")
        modelo_rf_8 = Criar_modelo_rf_8(base,caminho+ticker_split[0]+"_RF_8.joblib")
        modelo_rf_9 = Criar_modelo_rf_9(base,caminho+ticker_split[0]+"_RF_9.joblib")
        modelo_rf_10 = Criar_modelo_rf_10(base,caminho+ticker_split[0]+"_RF_10.joblib")

        #Modelos com o gaussiano mais sensíveis
        modelo_rf_11 = Criar_modelo_rf_11(base,caminho+ticker_split[0]+"_RF_11.joblib")
        modelo_rf_12 = Criar_modelo_rf_12(base,caminho+ticker_split[0]+"_RF_12.joblib")
        modelo_rf_13 = Criar_modelo_rf_13(base,caminho+ticker_split[0]+"_RF_13.joblib")
        modelo_rf_14 = Criar_modelo_rf_14(base,caminho+ticker_split[0]+"_RF_14.joblib")
        modelo_rf_15 = Criar_modelo_rf_15(base,caminho+ticker_split[0]+"_RF_15.joblib")
        modelo_rf_16 = Criar_modelo_rf_16(base,caminho+ticker_split[0]+"_RF_16.joblib")
        modelo_rf_17 = Criar_modelo_rf_17(base,caminho+ticker_split[0]+"_RF_17.joblib")
        modelo_rf_18 = Criar_modelo_rf_18(base,caminho+ticker_split[0]+"_RF_18.joblib")
        modelo_rf_19 = Criar_modelo_rf_19(base,caminho+ticker_split[0]+"_RF_19.joblib")
        
        