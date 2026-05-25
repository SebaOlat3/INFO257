# Contenido de models.py
import torch
import torch.nn as nn

# Arquitectura 1: Muy Simple (Capa de entrada -> Capa de salida directa)
class MLPClasificadorSimple(nn.Module):
    def __init__(self, input_dim, output_dim=2):
        super(MLPClasificadorSimple, self).__init__()
        self.output = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.output(x)

# Arquitectura 2: Media (1 Capa oculta con activación ReLU)
class MLPClasificadorMedio(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=2):
        super(MLPClasificadorMedio, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

# Arquitectura 3: Compleja (2 Capas ocultas con Dropout para evitar sobreajuste)
class MLPClasificadorComplejo(nn.Module):
    def __init__(self, input_dim, hidden_1=32, hidden_2=16, output_dim=2):
        super(MLPClasificadorComplejo, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_1)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Apaga 20% de neuronas aleatoriamente
        self.layer2 = nn.Linear(hidden_1, hidden_2)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(hidden_2, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.output(x)
        return x