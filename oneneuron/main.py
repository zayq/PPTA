# Doc peu descriptive pour linstant.

# Initialisation du dataset

dataset = [
    [0, 0],
    [1, -3],
    [2, -6],
    [3, -9],
    [4, -12],
]


# Function qui retourne le 'cost' du model selon sa weight. Plus la fonction retourne un nombre se rapprochant de zero plus notre weight sera bonne
def getCost(w):
    total_cost = 0
    for set in dataset:
        output = set[0] * w
        difference = set[1] - output
        cost = difference*difference
        total_cost += cost
    
    average_cost = total_cost / len(dataset)
    return average_cost


# Function qui est execute lorsque main.py se lance (Faire x iteration ou on ajuste un peu la weight dans le bon sens en fonction de la slope)
def main():
    
    w = 10
    rate = 0.002
    
    for x in range(10000):
        
        
        print(getCost(w))
        
        slope = (getCost(w - 0.01) - getCost(w)) / 0.01
        
        w += slope*rate
        
    print(w)
        
        
if __name__ == "__main__":
    main()
    
    
    
 
 
 
 
 
 
 
