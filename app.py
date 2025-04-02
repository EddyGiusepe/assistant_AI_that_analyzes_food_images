#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Este projeto foi inspirado no Tutorial do YouTube do
"Prof. Fabio Santos".

app.py
======
Nesta aplicação, o usuário carrega uma imagem de comida
e o app analisa a imagem e retorna uma descrição em texto
da imagem e uma análise nutricional, classificando a comida
como saudável ou não, e oferecendo sugestões de como tornar
a refeição mais saudável ou mais balanceada.

Run:
    streamlit run app.py
"""
import streamlit as st
import base64
import logging
from src.food_image_analyzer import FoodImageAnalyzer
from config.settings import GROQ_API_KEY
from utils.ansi_color_constants import *


def setup_logging() -> None:
    """Configures the logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("LOGs_foodbot.log"),
            logging.StreamHandler(),
        ],
    )


setup_logging()
logger = logging.getLogger(__name__)


# Streamlit app:
def main() -> None:
    logger.info(f"{GREEN}Iniciando a aplicação ...{RESET}")
    """
    Função principal da aplicação que inicializa o analisador de imagens de alimentos 
    e configura a interface do Streamlit.
    
    Esta função:
    1. Cria uma instância do FoodImageAnalyzer com os modelos apropriados da Groq
    2. Configura a barra lateral com uploader de imagens e informações do autor
    3. Configura o layout principal da aplicação com título e descrição
    4. Processa a imagem carregada, convertendo-a em base64
    5. Analisa a imagem e exibe os resultados da análise nutricional
    
    Returns:
        None
    """
    food_analyzer = FoodImageAnalyzer(
        GROQ_API_KEY,
        groq_vision_model="llama-3.2-11b-vision-preview",
        groq_text_model="llama-3.3-70b-versatile",
    )
    logger.info(f"{GREEN}Analisador de imagens de alimentos inicializado com sucesso.{RESET}")

    # Sidebar para carregar imagem e informações do Dr. Eddy Giusepe Chirinos Isidro:
    with st.sidebar:
        st.header("Carregar Imagem")
        uploaded_file = st.file_uploader(
            "Escolha uma imagem:", type=["jpg", "jpeg", "png"]
        )

        st.header("Informações do Autor")
        st.write(":blue[Senior Data Scientist: Dr. Eddy Giusepe Chirinos Isidro]")
        st.markdown(
            "[LinkedIn](https://www.linkedin.com/in/eddy-giusepe-chirinos-isidro-phd-85a43a42/)"
        )
        st.write("📧 e-mail: eddychirinos_unac@hotmail.com")
        st.write("🌐 GitHub: [Hugging Face](https://huggingface.co/EddyGiusepe)")

    # Conteúdo principal da página:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./images/imageBot.png", width=200)
    st.title(":blue[Analisador de Alimentos em Imagens]", anchor="center")
    st.markdown(
        """<span style="color:pink">FoodBot: seu assistente inteligente para análise de alimentos em imagens.
             Ele é capaz de identificar ingredientes e fornecer informações nutricionais
             para ajudá-lo a tomar decisões mais saudáveis.</span>""",
        unsafe_allow_html=True,
    )

    if uploaded_file is not None:
        logger.info(f"{GREEN}Imagem carregada com sucesso.{RESET}")
        st.image(uploaded_file, caption="Imagem carregada", width=400)
        # To read file as bytes:
        bytes_data = uploaded_file.read()
        logger.info(f"{GREEN}Convertendo imagem para base64 . . .{RESET}")
        base64_image = base64.b64encode(bytes_data).decode("utf-8")
        # Mostra mensagem de carregamento durante o processamento:
        with st.spinner("Analisando a imagem ..."):
            logger.info(f"{GREEN}Iniciando a análise da imagem . .  .{RESET}")
            prompt = """
                     Analise e descreva com detalhes a seguinte imagem, incluindo a aparência do 
                     objeto(s). Nota: Sempre responda em português do Brasil (pt-br).
                     """
            image_description = food_analyzer.image_to_text(base64_image, prompt)
            logger.info(f"{GREEN}Descrição da imagem obtida com sucesso.{RESET}")
            food_description = food_analyzer.analyze_food(image_description)
            logger.info(f"{GREEN}Análise nutricional finalizada com sucesso.{RESET}")
        with st.expander("Análise finalizada do alimento", expanded=True):
            st.write(food_description)


if __name__ == "__main__":
    main()
