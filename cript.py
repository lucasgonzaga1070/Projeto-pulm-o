str = input()


# encript --------------------------------------------------------------------------------------------------------------
def encript(strg):
    fator = 2 * len(strg)
    mini = 33
    maxi = 125
    numbers = [ord(c) for c in strg]
    newNumbers = []
    for number in numbers:
        newNumber = number - fator
        while newNumber > maxi or newNumber < mini:
            if newNumber > maxi:
                newNumber -= (maxi - mini + 1)
            elif newNumber < mini:
                newNumber = maxi - (mini - newNumber - 1)

        newNumbers.append(newNumber)

    return ''.join([chr(n) for n in newNumbers])


# decript --------------------------------------------------------------------------------------------------------------
def decript(newStrg):
    fator = 2 * len(newStrg)
    mini = 33
    maxi = 125
    otherNumbers = [ord(c) for c in newStrg]
    newOtherNumbers = []
    for otherNumber in otherNumbers:
        newOtherNumber = otherNumber + fator
        while newOtherNumber > maxi or newOtherNumber < mini:
            if newOtherNumber > maxi:
                newOtherNumber -= (maxi - mini + 1)
            elif newOtherNumber < mini:
                newOtherNumber = maxi - (mini - newOtherNumber - 1)

        newOtherNumbers.append(newOtherNumber)

    return ''.join([chr(n) for n in newOtherNumbers])


a = encript(str)
b = decript(a)

print(str)
print(a)
print(b)
