"""Motor de Transcrição de Imagens para Texto
Este módulo contém tanto o motor antigo (TrOCR) quanto o novo motor de alta velocidade (Gemini 1.5 Flash).
"""

import os
import json
import base64
from typing import Optional, Tuple
from pathlib import Path
import logging
from dotenv import load_dotenv

# Importações para o motor antigo (TrOCR)
from job_client import upload_image as client_upload_image
from job_client import start_job_from_server_path, poll_job

# Importações para o novo motor (Gemini)
import google.generativeai as genai
from PIL import Image

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()

class TranscriptionEngine:
    """Classe principal para gerenciar motores de transcrição"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = None
        self._init_gemini()
    
    def _init_gemini(self):
        """Inicializa o cliente Gemini"""
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                # Use um nome de modelo suportado pela Generative Language API pública
                # Evitar aliases que podem mapear para versões Vertex (ex.: "-002")
                model_name = 'gemini-2.5-flash'  # escolhido por disponibilidade pública e suporte multimodal
                self.gemini_model = genai.GenerativeModel(model_name)
                logger.info(f"✅ Gemini inicializado com sucesso (modelo: {model_name})")
            except Exception as e:
                logger.error(f"❌ Erro ao inicializar Gemini: {e}")
                self.gemini_model = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY não encontrada - modo fallback")
    
    def validate_gemini_credentials(self) -> bool:
        """Valida se as credenciais do Gemini estão funcionando"""
        if not self.gemini_model:
            return False
        
        try:
            # Teste simples com texto
            response = self.gemini_model.generate_content("Diga apenas 'OK' se você conseguir me ouvir.")
            ok = bool(response and getattr(response, 'text', '') and 'OK' in response.text)
            if not ok:
                logger.error(f"❌ Validação retornou resposta inesperada: {getattr(response, 'text', '')}")
            return ok
        except Exception as e:
            logger.error(f"❌ Falha na validação do Gemini: {e}")
            return False


# ===============================
# MOTOR ANTIGO (TrOCR - DIESEL)
# ===============================

def transcrever_imagem_com_trocr(image_bytes: bytes, language: str = "English") -> str:
    """
    Motor antigo: Transcreve imagem usando TrOCR através do serviço externo.
    
    Args:
        image_bytes: Bytes da imagem a ser transcrita
        language: Idioma para transcrição
    
    Returns:
        str: Texto transcrito ou string vazia em caso de erro
    """
    logger.info("🐌 Iniciando transcrição com TrOCR (motor lento)...")
    
    try:
        # 1. Salvar temporariamente
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        # 2. Upload para o serviço
        server_path = client_upload_image(temp_path)
        if not server_path:
            logger.error("❌ Falha no upload para o serviço TrOCR")
            return ""
        
        # 3. Iniciar job
        job_id = start_job_from_server_path(server_path, language)
        if not job_id:
            logger.error("❌ Falha ao iniciar job no serviço TrOCR")
            return ""
        
        # 4. Aguardar resultado
        final_status = poll_job(job_id)
        if not final_status or final_status.get('status') != 'completed':
            logger.error("❌ Job TrOCR falhou ou expirou")
            return ""
        
        detected_text = final_status.get('result', '')
        logger.info(f"✅ TrOCR concluído: {len(detected_text)} caracteres transcritos")
        
        # 5. Limpar arquivo temporário
        os.unlink(temp_path)
        
        return detected_text
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição TrOCR: {e}")
        return ""


# ===============================
# MOTOR NOVO (GEMINI - ELÉTRICO)
# ===============================

def transcrever_imagem_com_gemini(image_bytes: bytes) -> str:
    """
    Motor novo: Transcreve imagem usando Gemini 1.5 Flash de alta velocidade.
    
    Args:
        image_bytes: Bytes da imagem a ser transcrita
    
    Returns:
        str: Texto transcrito ou string vazia em caso de erro
    """
    logger.info("⚡ Iniciando transcrição com Gemini 1.5 Flash (motor elétrico)...")
    
    try:
        # 1. Validar credenciais
        engine = TranscriptionEngine()
        if not engine.gemini_model:
            logger.error("❌ Gemini não está disponível")
            return ""
        
        # 2. Converter bytes para imagem PIL
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes))
        
        # 3. Criar prompt otimizado para OCR
        prompt_ocr = """A sua única tarefa é atuar como um sistema de OCR (Reconhecimento Ótico de Caracteres) de alta precisão. 

Transcreva todo o texto, incluindo números e símbolos, presente na imagem fornecida. 

Devolva apenas o texto transcrito, sem qualquer comentário, formatação adicional ou explicação.

Se não houver texto na imagem, responda apenas: "Nenhum texto detectado"."""

        # 4. Chamada multimodal
        response = engine.gemini_model.generate_content([prompt_ocr, image])
        
        # 5. Extrair texto da resposta
        transcribed_text = response.text.strip()
        
        logger.info(f"✅ Gemini concluído: {len(transcribed_text)} caracteres transcritos")
        return transcribed_text
        
    except Exception as e:
        logger.error(f"❌ Erro na transcrição Gemini: {e}", exc_info=True)
        return ""


# ===============================
# FUNÇÃO DE TESTE ISOLADO  
# ===============================

def testar_motor_gemini(image_path: str) -> bool:
    """
    Testa o motor Gemini de forma isolada com uma imagem de exemplo.
    
    Args:
        image_path: Caminho para imagem de teste
    
    Returns:
        bool: True se o teste passou, False caso contrário
    """
    logger.info("🧪 Testando motor Gemini de forma isolada...")
    
    try:
        # Ler imagem de teste
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Testar transcrição
        resultado = transcrever_imagem_com_gemini(image_bytes)
        
        if resultado and resultado != "Nenhum texto detectado":
            logger.info(f"✅ Teste passou! Resultado: {resultado[:100]}...")
            return True
        else:
            logger.warning("⚠️ Teste retornou resultado vazio")
            return False
            
    except Exception as e:
        logger.error(f"❌ Teste falhou: {e}")
        return False


if __name__ == "__main__":
    # Testar validação de credenciais
    engine = TranscriptionEngine()
    if engine.validate_gemini_credentials():
        print("✅ Credenciais Gemini validadas com sucesso!")
    else:
        print("❌ Falha na validação das credenciais Gemini")
    
    # Testar com imagem se disponível
    test_image = "imagem.jpg"
    if os.path.exists(test_image):
        if testar_motor_gemini(test_image):
            print("✅ Motor Gemini está funcionando perfeitamente!")
        else:
            print("❌ Motor Gemini apresentou problemas")
    else:
        print(f"⚠️ Imagem de teste não encontrada: {test_image}")