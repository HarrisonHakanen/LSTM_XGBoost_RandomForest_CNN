import time



import funcoes_random_forest as funcoes

tickers=["MDIA3.SA","BRFS3.SA","KLBN11.SA","SUZB3.SA",
         "B3SA3.SA","ITUB4.SA","BBDC3.SA","ENGI11.SA",
         "TAEE11.SA","ELET6.SA","EGIE3.SA","CSAN3.SA",
         "RRRP3.SA","TEND3.SA","MOVI3.SA","INTB3.SA",
         "GMAT3.SA","VIVA3.SA","LREN3.SA","SOMA3.SA",
         "RAIL3.SA","IGTI11.SA","ABEV3.SA","CRFB3.SA",
         "ARZZ3.SA","CYRE3.SA","PETR3.SA"]


faltando = ["GMAT3.SA","VIVA3.SA","LREN3.SA","SOMA3.SA",
         "RAIL3.SA","IGTI11.SA","ABEV3.SA","CRFB3.SA",
         "ARZZ3.SA","CYRE3.SA","PETR3.SA"]

faltando2 = ["SUZB3.SA","EGIE3.SA","LREN3.SA"]

faltando3 = ["CYRE3.SA","ABEV3.SA","ELET3.SA","ENGI11.SA","ITUB4.SA","PETR3.SA","SOMA3.SA"]

tickers2=["KLBN11.SA","RAIL3.SA","VALE3.SA","CRFB3.SA","VIVT3.SA"]

tickers3 = ["ABEV3.SA","SOMA3.SA"]


#as ações INTB3.SA, IGTI11.SA estão com problemas
funcoes.Criar_modelos_random_forest(tickers3,0)
