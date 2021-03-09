# imports

import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
# import TeachableMachinePrediction

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    image = Image.open('fliptec.jpg')   # abre imagem com logo
    st.image(image)                     # mostra imagem

    # mostra texto
    st.write("<h3>Arraste uma imagem no <font color='#27d5e2'>BOX abaixo</font>, para indentificar o componente.</h3>",unsafe_allow_html = True)

    # mostra input de upload de arquivos
    uploaded_file = st.file_uploader("")

    # se algum arquivo for arrastado ou aberto
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        #width, height (image.size)

        # Load the model
        model = tensorflow.keras.models.load_model('keras_model.h5')

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # display the resized image
        st.image(image)

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)

        # pega o resultado mais relevante encontrado pelo modelo
        resultado = np.amax(prediction) * 100                          # pega o maior valor e multiplica por 100 para pegar percentual
        resultado = np.format_float_positional(np.float16(resultado))  # arredonda o número

        # pega o indice do maior resultado
        resultado_indice = np.argmax(prediction)

        # le arquivo com os tipos de componentes cadastrados
        labelsfile = open("labels.txt", 'r')

        # lê todos os itens e colocar na matriz "classes"
        classes = []
        line = labelsfile.readline()
        while line:
            # retrieve just class name and append to classes
            classes.append(line.split(' ', 1)[1].rstrip())
            line = labelsfile.readline()
        # close label file
        labelsfile.close()

        # mostra na tela o tipo do componente encontrado e o percentual
        st.write("<h1>" + classes[resultado_indice] + ": <font color='#27d5e2'>" + resultado + "&percnt;</font></h1>",
             unsafe_allow_html=True)


