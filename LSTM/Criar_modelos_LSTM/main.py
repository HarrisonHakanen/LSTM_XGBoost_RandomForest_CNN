import time

import funcoes as fc

tickers=[["MDIA3.SA",80,0.003],["BRFS3.SA",80,0.0037],["KLBN11.SA",80,0.003],["SUZB3.SA",80,0.004],
         ["B3SA3.SA",80,0.0045],["ITUB4.SA",80,0.0045],["BBDC3.SA",80,0.0045],["ENGI11.SA",80,0.0045],
         ["TAEE11.SA",80,0.0045],["ELET6.SA",80,0.003],["EGIE3.SA",80,0.0037],["CSAN3.SA",80,0.0037],
         ["RRRP3.SA",80,0.0037],["TEND3.SA",80,0.0037],["MOVI3.SA",80,0.0045],["INTB3.SA",80,0.0045],
         ["GMAT3.SA",80,0.0045],["VIVA3.SA",80,0.003],["LREN3.SA",80,0.004],["SOMA3.SA",80,0.004],
         ["RAIL3.SA",80,0.003],["IGTI11.SA",80,0.003],["ABEV3.SA",80,0.004],["CRFB3.SA",80,0.0027],
         ["ARZZ3.SA",80,0.0035],["CYRE3.SA",80,0.0035],["PETR3.SA",80,0.007]]

tickers_1 = [["MDIA3.SA",80,0.003],["BRFS3.SA",80,0.0037],["KLBN11.SA",80,0.003],["SUZB3.SA",80,0.004]]
tickers_2 = [["B3SA3.SA",80,0.0045],["ITUB4.SA",80,0.0045],["BBDC3.SA",80,0.0045],["ENGI11.SA",80,0.0045],["TAEE11.SA",80,0.0045],["ELET6.SA",80,0.003]]
tickers_3 = [["EGIE3.SA",80,0.0037],["CSAN3.SA",80,0.0037],["RRRP3.SA",80,0.0037],["TEND3.SA",80,0.0037],["MOVI3.SA",80,0.0045],["INTB3.SA",80,0.0045]]
tickers_5 = [["GMAT3.SA",80,0.0045],["VIVA3.SA",80,0.003],["LREN3.SA",80,0.004],["SOMA3.SA",80,0.004],["RAIL3.SA",80,0.003],["IGTI11.SA",80,0.003]]
tickers_7 = [["ARZZ3.SA",80,0.0035],["CYRE3.SA",80,0.0035],["PETR3.SA",80,0.007],["ABEV3.SA",80,0.004],["CRFB3.SA",80,0.0027]]

tickers_8 = [["SOMA3.SA",80,0.004],["ABEV3.SA",80,0.004]]
inicio = time.time()

fc.Criar_modelos_gaussian(tickers_8,0)

fim = time.time()

print("\n")
print("\n")
print(fim-inicio)