# f=open('data_no','r')
# cities=[]
#cities = [line.strip('\n') for line in open('data_no') ]
#print(cities)
with open('data_no','r') as f:
    cities = f.read()

city = cities.splitlines()
print(city)
