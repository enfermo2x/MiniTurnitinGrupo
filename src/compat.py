"""Comprobaciones de disponibilidad de dependencias opcionales.

Funciones pequeñas para detectar si paquetes como spaCy, torch, transformers
o sentence-transformers están instalados en el entorno. Usadas por la UI
para mostrar el estado y habilitar opciones condicionales.
"""
import importlib.util


def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def has_spacy() -> bool:
    return _is_installed("spacy")


def has_sentence_transformers() -> bool:
    return _is_installed("sentence_transformers")


def has_torch() -> bool:
    return _is_installed("torch")


def has_transformers() -> bool:
    return _is_installed("transformers")


def has_gpt2() -> bool:
    """Verifica si `transformers` y las clases GPT-2 están disponibles.

    Devuelve False si `transformers` o `torch` no están instalados o si importar
    las clases específicas falla.
    """
    if not has_transformers():
        return False
    try:
        # Intentar importar las clases relevantes sin forzar descarga de pesos
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast  # type: ignore
        return True
    except Exception:
        return False
