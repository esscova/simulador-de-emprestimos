# --- modulos
import os
import joblib
import logging
import streamlit as st 

# --- configs
logger = logging.getLogger(__name__)

# --- funcoes
@st.cache_resource
def load_prediction_models(model_files_dict: dict) -> dict:
    """
        Carrega modelos de previsão que suportam predict_proba.
    """
    loaded_models = {}
    logger.info("Iniciando carregamento dos modelos de previsão...")
    
    try:
        for name, filepath in model_files_dict.items():
            if os.path.exists(filepath):
                model = joblib.load(filepath)
                if hasattr(model, "predict_proba"):
                    loaded_models[name] = model
                    logger.info(f"Modelo '{name}' carregado com sucesso de {filepath}.")
                else:
                    logger.warning(f"Modelo '{name}' em {filepath} não suporta 'predict_proba'. Será ignorado.")
            else:
                logger.error(f"Arquivo do modelo '{name}' não encontrado em {filepath}.")
                st.error(f"Erro Crítico: Arquivo do modelo '{name}' não encontrado.")

        if not loaded_models:
            logger.critical("Nenhum modelo válido com 'predict_proba' foi carregado.")
            st.error("Falha ao carregar modelos válidos para previsão.")
            return None
        logger.info(f"{len(loaded_models)} modelos carregados com sucesso.")
        return loaded_models

    except Exception as e:
        logger.error(f"Exceção durante o carregamento dos modelos: {e}", exc_info=True)
        st.error(f"Erro fatal ao carregar modelos: {e}")
        return None