#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro
"""
from groq import Groq
import base64


class FoodImageAnalyzer:
    """
    Classe para análise de imagens de alimentos usando a API Groq.

    Attributes:
        client (Groq): Cliente da API Groq.
        llama32_11b_vision (str): Modelo para análise de imagens.
        llama33_70b_versatile (str): Modelo para análise de texto.
        logger (logging.Logger): Logger para registro de eventos.
    """

    def __init__(
        self,
        groq_api_key: str,
        groq_vision_model: str = "llama-3.2-11b-vision-preview",
        groq_text_model: str = "llama-3.3-70b-versatile",
    ):
        """
        Inicializa o FoodImageAnalyzer.

        Args:
            groq_api_key (str): Chave da API Groq.
            groq_vision_model (str): Modelo para análise de imagens.
            groq_text_model (str): Modelo generativo só de texto.
        """
        self.client = Groq(api_key=groq_api_key)
        self.groq_vision_model = groq_vision_model  # Multimodal
        self.groq_text_model = groq_text_model  # Só texto

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Codifica uma imagem em base64.

        Args:
            image_path (str): Caminho do arquivo de imagem.

        Returns:
            str: Imagem codificada em base64.

        Raises:
            FileNotFoundError: Se o arquivo não for encontrado.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_to_text(self, base64_image: str, prompt: str) -> str:
        """
        Realiza uma análise de uma imagem e retorna uma descrição textual.

        Args:
            base64_image (str): Imagem codificada em base64.
            prompt (str): Instrução para análise da imagem.

        Returns:
            str: Descrição textual da imagem.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=self.groq_vision_model,
            # max_tokens=4096,
            temperature=0,
        )
        return chat_completion.choices[0].message.content

    def analyze_food(self, image_description: str) -> str:
        """
        Analisa a descrição do alimento e gera recomendações nutricionais.

        Args:
            image_description (str): Descrição da imagem do alimento.

        Returns:
            str: Análise nutricional e recomendações.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""Você é um especialista em nutrição e alimentação. Você analisa a descrição, fornecida, de uma imagem de alimentos.
                                   Então, você deve fornecer ao usuário observações sobre a descrição da imagem, como: possíveis calorias,
                                   classificação do alimento (se é bom ou ruim), e oferece sugestões de como tornar a refeição mais saudável
                                   ou mais balanceada. NOTE: Sempre responda em português (pt-br).""",
                },
                {"role": "user", "content": image_description},
            ],
            model=self.groq_text_model,
        )
        return chat_completion.choices[0].message.content
