from BAG_WORDS3 import ai
 
modelo = ai(arquivo="BAG_WORDS/4_classes_ESPE.json", file_words="BAG_WORDS/4_Classes_ESPE_WORDS.txt", file_classes="BAG_WORDS/4_Classes", file_model="modelo_4_Classes_ESPE3.h5", training=True)

modelo.load()

inputs, outputs = modelo.load_training()  # nesse json temos uma lista de palavras chamadas "input", eu quero que vc pegue as palavras dos inputs e escreve todas as palavras unicas NÃO REPETIDAS contidas neles:, NÃO QUERO CODIGO, APENAS UMA LISTA DAS PALAVRAS:
                                         # acima o prompt para o chatGPT dar todas as palavras unicas do JSON

model, trained_model = modelo.create_model(inputs, outputs)

model.save('modelo_4_Classes_ESPE3.h5')