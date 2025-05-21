# FIAPChallenge4

Este projeto realiza detecção de emoções em tempo real utilizando a webcam, com auxílio da biblioteca [DeepFace](https://github.com/serengil/deepface) e OpenCV.

## Funcionalidades

- Captura de vídeo pela webcam.
- Detecção de faces e análise de emoções.
- Exibição das emoções detectadas em português sobre cada rosto identificado.

## Requisitos

- Python 3.10
- As dependências estão listadas em [requirements.txt](requirements.txt).

## Instalação

## 1. Crie e ative um ambiente virtual (opcional, mas recomendado):

   ```sh
   python -m venv venv-py310
   source venv-py310/Scripts/activate  # Windows

## 2. Instale as dependências:
   pip install -r requirements.txt

## 3. Executar o script principal
python main.py

A janela de vídeo será aberta mostrando as emoções detectadas em tempo real. Pressione ESC para sair.
