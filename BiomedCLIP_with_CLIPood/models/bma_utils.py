import copy

def init_bma_model(model):
    return copy.deepcopy(model)

def update_bma_model(model, bma_model, step, total_steps):
    beta = step / total_steps
    for p, p_bma in zip(model.parameters(), bma_model.parameters()):
        p_bma.data = beta * p_bma.data + (1 - beta) * p.data
