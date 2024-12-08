import numpy as np
class ativaDegrau:
    @staticmethod
    def activate(x):
        # Para um único valor
        return 1 if x >= 0 else -1

    @staticmethod
    def activate_array(x):
        # Para um array de valores, utiliza a vetorização
        return np.where(x >= 0, 1, -1)  # Retorna 1 onde x >= 0 e -1 onde x < 0

