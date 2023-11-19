# WORK IN PROGRESS
# Aucune documentation car je nai pas finis


from typing import List
from getData import readCSV

wines = readCSV()


class Input:
    
    def __init__(self, name: str, value: float) -> None:
        
        self.name = name
        self.value = value

class InputLayers:

    def __init__(self, inputs: List[Input]) -> None:
        
        self.inputs = inputs
    
    def calculateCost(self, b, InputWeights) -> float:
        total_cost = 0
        
        for wine in wines:
            
            quality = wine['quality'] 
            expected_value = -1 if quality == 'bad' else 1
            
            current_value = 0
            
            for input in self.inputs:
                current_value += input.value*InputWeights[input.name]
            
            current_value += b
            
            d = current_value - expected_value
            cost = d*d
            total_cost += cost
        
        
        return total_cost/len(wines)






def Train():
    

    rate = 0.001
    b = 0.001
    InputWeights = {
    'fixed acidity': 1,
    'volatile acidity': 1,
    'citric acid': 1,
    'residual sugar': 1,
    'chlorides': 1,
    'free sulfur dioxide': 1,
    'total sulfur dioxide': 1,
    'density': 1,
    'pH': 1,
    'sulphates': 1,
    'alcohol': 0.1,
    }

    for x in range(100):
            wine_inputs = []
        
            for name, value in wines[0].items():
                if name != 'quality':
                    input_obj = Input(name, value)
                    wine_inputs.append(input_obj)

            inputLayers = InputLayers(wine_inputs)
            cost = inputLayers.calculateCost(b=b, InputWeights=InputWeights)

            TempWeights = InputWeights.copy()
            for name, value in wines[0].items():
                if name != 'quality':
                    DerivatedInputWeights = InputWeights.copy()
                    DerivatedInputWeights[name] += 0.01
                    derivated_cost = inputLayers.calculateCost(b=b, InputWeights=DerivatedInputWeights)
                    

                    slope = (derivated_cost - cost) / 0.01
                    TempWeights[name] -= slope*rate
                    
                
            derivated_cost_for_b = (inputLayers.calculateCost(b=b + 0.01, InputWeights=InputWeights) - cost) / 0.01
            b -= derivated_cost_for_b*rate
            
            InputWeights = TempWeights
            
            
            print(inputLayers.calculateCost(b=b, InputWeights=InputWeights))
    cost = inputLayers.calculateCost(b=b, InputWeights=InputWeights)

    return InputWeights, b


InputWeights, b = Train()

print(InputWeights, b)

def Test():
    bad_count = 0

    for wine, data in enumerate(wines):
        result = 0  

        for name, value in data.items():
            if name in InputWeights:
                result += value * InputWeights[name]


        result += b

        classification = 'good' if result > 0 else 'bad'

        if classification != data['quality']:
            bad_count += 1

        print(f"Sample {wine}: {classification}")

    print("Number of misclassifications:", bad_count)

Test()