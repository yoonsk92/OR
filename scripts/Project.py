import pulp

model = pulp.LpProblem("The Ten Spot", pulp.LpMinimize)

delta_positive = {}
delta_negative = {}

for s in range(3):
    for d in range(7):        
        delta_positive[s,d] = pulp.LpVariable("delta_positive(%s,%s)" % (s,d), lowBound = 0, cat = "Continuous")
        delta_negative[s,d] = pulp.LpVariable("delta_negative(%s,%s)" % (s,d), lowBound = 0, cat = "Continuous")

y = {}
x = {}
for n in range(8):
    for s in range(3):
        for d in range(7):
            y[n,s,d] = pulp.LpVariable("y(%s,%s,%s)" % (n,s,d), cat = "Binary")
            x[n,s,d] = pulp.LpVariable("x(%s,%s,%s)" % (n,s,d), lowBound = 0, cat = "Integer")

p = [0.052, 0.037, 0.033, 0.029, 0.026, 0.039, 0.055]

#objective
model += 31 * pulp.lpSum(p[d] * (delta_negative[s,d] + delta_positive[s,d]) for d in range(7) for s in range(3))

for n in range(8):
    for d in range(7):
        model += y[n,0,d] + y[n,2,d] <= 1

for n in range(8):
    model += pulp.lpSum(y[n,s,d] for d in range(7) for s in range(3)) <= 11 

for n in range(8):
    for d in range(7):
        model += pulp.lpSum(y[n,s,d] for s in range(3)) <= 2

for d in range(7):
    for s in range(3):
        model += pulp.lpSum(y[n,s,d] for n in range(8)) <= 8
        model += pulp.lpSum(y[n,s,d] for n in range(8)) >= 2

for n in range(8):
    for s in range(3):
        for d in range(7):
            model += x[n,s,d] >= 2 * y[n,s,d]
            model += x[n,s,d] <= 6 * y[n,s,d]
            #model += x[n,s,d] <= 5 * y[n,s,d]

A = [[6,7,9,11,12,14,8],[11,13,16,19,21,24,14],[7,9,11,12,14,16,9]]
W = [[0.255,0.255,0.255,0.255,0.255,0.511,0.255],[0.447,0.447,0.447,0.447,0.447,0.895,0.447],[0.297,0.297,0.297,0.297,0.297,0.594,0.297]]

for s in range(3):
    for d in range(7):
        model += (pulp.lpSum(x[n,s,d] for n in range(8)) + delta_negative[s,d] - delta_positive[s,d] == A[s][d] + W[s][d])

model.solve()
pulp.value(model.objective)
for variable in model.variables():
    print ("{} = {}".format(variable.name, variable.varValue))

print("Our model status is..", pulp.LpStatus[model.status], '\nOptimal value is..', pulp.value(model.objective))





