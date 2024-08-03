from enum import Enum

class Animal(Enum):
    dog = 1
    cat = 2
    lion = 1


# Comparison using "is"
# if Animal.dog is Animal.cat:
#     print("Dog and cat are same animals")
# else:
#     print("Dog and cat are different animals")
#
# # Comparison using "!="
# if Animal.lion != Animal.cat:
#     print("Lions and cat are different")
# else:
#     print("Lions and cat are same")


ll = [Animal.dog, Animal.cat, Animal.lion]

for l in ll:
    print (l.value, "\t", l)
    if (l is Animal.dog):
        print ("Bark")
    if (l.value == 1):
        print ("Woof")
