from BAG_WORDS3 import ai
 
modelo = ai(arquivo="BAG_WORDS\data_sem_padrao.json", file_words="BAG_WORDS\words2.txt", file_classes="BAG_WORDS\classes.txt", file_model="modelo6.h5", training=True)

modelo.load()

inputs, outputs = modelo.load_training()


model, trained_model = modelo.create_model(inputs, outputs)

model.save('BAG_WORDS\modelos\modelo6.h5')