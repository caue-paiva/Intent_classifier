from BAG_WORDS3 import ai

modelo=  ai(arquivo="BAG_WORDS/4_classes_ESPE.json", file_words="BAG_WORDS/4_Classes_ESPE_WORDS.txt", file_classes="BAG_WORDS/4_Classes", file_model="modelo_4_Classes_ESPE3.h5", training=False)


modelo.load_files()
frase = "quais as princiais causas da revolução francesa?"   # me de uma questao do enem 
frase = frase.lower()
resposta= modelo.get_classe(frase) #testar no codigo de routing uma probabilidade de 0.98 ou mais para ser aceito.

print(frase + "\n \n")
print(resposta)
probabilidade = float(resposta['probabilidade'])
# print(probabilidade)
# if probabilidade >= 0.98:
#     print("passou")    template para a verificação de precisão feita no codigo de routing