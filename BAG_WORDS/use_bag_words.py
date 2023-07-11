from BAG_WORDS3 import ai

modelo=  ai(arquivo="BAG_WORDS\data_sem_padrao.json", file_words="BAG_WORDS\words2.txt", file_classes="BAG_WORDS\classes.txt", file_model="BAG_WORDS\modelos\modelo6.h5", training=False)


modelo.load_files()
frase = "eu quero uma questão do ENEM sobre a colonização do brasil"
frase = frase.lower()
resposta= modelo.get_classe(frase)

print(resposta)