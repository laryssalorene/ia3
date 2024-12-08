import os
import numpy as np
import cv2

class LerImagem:
    
    @staticmethod
    def obter_dados(img_size=50):
        raiz = "RecFac"
        # Usando os.listdir e list comprehension para pegar todas as pastas
        pastas_pessoas = [os.path.join(raiz, pasta) for pasta in os.listdir(raiz) if os.path.isdir(os.path.join(raiz, pasta))]

        C = 20  # Total de classes (assumindo que há 20 pastas, uma por pessoa)
        X = []  # Lista para armazenar as imagens vetorizadas
        Y = []  # Lista para armazenar os rótulos (one-hot encoding)

        # Iterar pelas pastas (pessoas)
        for i, pasta in enumerate(pastas_pessoas):
            imagens_pasta = os.listdir(pasta)  # Lista de imagens dentro da pasta da pessoa

            for imagem_nome in imagens_pasta:
                path_imagem = os.path.join(pasta, imagem_nome)

                # Carrega a imagem em escala de cinza
                imagem = cv2.imread(path_imagem, cv2.IMREAD_GRAYSCALE)
                # Redimensiona a imagem
                imagem_redimensionada = cv2.resize(imagem, (img_size, img_size))

                # Vetorização da imagem (transforma em vetor 1D)
                imagem_vetorizada = imagem_redimensionada.flatten()

                # Armazenando a imagem vetorizada na lista X
                X.append(imagem_vetorizada)

                # Criando o vetor de rótulos one-hot encoding
                rotulo = np.zeros(C)  # Inicializa todos os valores com 0
                rotulo[i] = 1  # Marca a classe correta como 1
                Y.append(rotulo)

        # Convertendo as listas para arrays numpy
        X = np.array(X)  # Forma (n_amostras, n_features)
        Y = np.array(Y)  # Forma (n_amostras, n_classes)

        return X, Y
