# Doc peu descriptive pour linstant.

# Initialisation du dataset



dataset = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]



# Function qui retourne le 'cost' du model selon les deux weights + le biais. Plus la fonction retourne un nombre se rapprochant de zero plus nos parametres seront optimaux

def getCost(w1, w2, b):
    total_cost = 0
    for set in dataset:
        x1 = set[0]
        x2 = set[1]
        
        difference = x1*w1 + x2*w2 + b - set[2]
        cost = difference*difference
        total_cost += cost
    
    average_cost = total_cost / len(dataset)
    return average_cost



# Function qui est execute lorsque main.py se lance (Faire x iteration ou on ajuste nos parametres dans le bon sens en fonction de la slope)
def main():
    
    
    w1 = 100
    w2 = 100
    b = 100
    rate = 0.2
    
    for x in range(1000):
        cost = getCost(w1, w2, b)
        w1slope = (getCost(w1 + 0.1, w2, b) - cost) / 0.1
        w2slope = (getCost(w1, w2 + 0.1, b) - cost) / 0.1
        bslope = (getCost(w1, w2, b + 0.1) - cost) / 0.1
        
        print(w1slope)
        
        w1 -= w1slope*rate
        w2 -= w2slope*rate
        b -= bslope*rate
        
    print(w1, w2, b)
    print(getCost(w1, w2, b=b))
    
        

if __name__ == "main.py":
    main()