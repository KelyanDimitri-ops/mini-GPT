import sys
import os
import math
import torch

# Ajoute le dossier racine au chemin d'importation Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autograd.value import Value


def test_autograd_vs_pytorch():
    """
    Compare le résultat forward et les gradients backward de Value
    par rapport à PyTorch sur un graphe d'opérations complexes.
    """
    # 1. Valeurs initiales
    a_val = -3.0
    b_val = 2.0
    c_val = 0.5

    # Instanciation pour Micrograd
    a_mg = Value(a_val, label='a')
    b_mg = Value(b_val, label='b')
    c_mg = Value(c_val, label='c')

    # Instanciation pour PyTorch (float64 pour maximiser la précision)
    a_pt = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)
    c_pt = torch.tensor(c_val, dtype=torch.float64, requires_grad=True)

    # 2. Forward Pass (Calcul de l'expression)
    # Micrograd
    e_mg = a_mg + b_mg
    f_mg = e_mg * c_mg
    g_mg = f_mg ** 2
    h_mg = g_mg.exp()
    i_mg = h_mg.log()
    j_mg = i_mg / (b_mg - c_mg)
    k_mg = j_mg.relu()
    res_mg = k_mg.tanh()

    # PyTorch
    e_pt = a_pt + b_pt
    f_pt = e_pt * c_pt
    g_pt = f_pt ** 2
    h_pt = torch.exp(g_pt)
    i_pt = torch.log(h_pt)
    j_pt = i_pt / (b_pt - c_pt)
    k_pt = torch.relu(j_pt)
    res_pt = torch.tanh(k_pt)

    # 3. Backward Pass
    res_mg.backward()
    res_pt.backward()

    # 4. Vérification numérique (seuil de tolérance tol = 1e-6)
    tol = 1e-6

    # Test du Forward
    assert math.isclose(res_mg.data, res_pt.item(), abs_tol=tol), \
        f"Erreur Forward: {res_mg.data} != {res_pt.item()}"

    # Test des Gradients (Backward)
    assert math.isclose(a_mg.grad, a_pt.grad.item(), abs_tol=tol), \
        f"Erreur Gradient sur 'a': {a_mg.grad} != {a_pt.grad.item()}"
    assert math.isclose(b_mg.grad, b_pt.grad.item(), abs_tol=tol), \
        f"Erreur Gradient sur 'b': {b_mg.grad} != {b_pt.grad.item()}"
    assert math.isclose(c_mg.grad, c_pt.grad.item(), abs_tol=tol), \
        f"Erreur Gradient sur 'c': {c_mg.grad} != {c_pt.grad.item()}"

    print("✅ TEST REUSSI : Micrograd et PyTorch donnent exactement les mêmes résultats !")
    print(f"Forward  => Value: {res_mg.data:.8f} | PyTorch: {res_pt.item():.8f}")
    print(f"Grad 'a' => Value: {a_mg.grad:.8f} | PyTorch: {a_pt.grad.item():.8f}")
    print(f"Grad 'b' => Value: {b_mg.grad:.8f} | PyTorch: {b_pt.grad.item():.8f}")
    print(f"Grad 'c' => Value: {c_mg.grad:.8f} | PyTorch: {c_pt.grad.item():.8f}")


if __name__ == '__main__':
    test_autograd_vs_pytorch()
