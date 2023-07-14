import numpy as np

def FC(L, M):
    "*** YOUR CODE HERE ***"

    # dictionary to store the assigned positions of the markers
    global countNodes_fc
    countNodes_fc  = 0
    positions = {}

    # Assign position -1 to keys 1....M
    for i in range(0, M):
        positions[i + 1] = -1

    # minimum length of optimal solution = M C 2 (i.e. number of possible combinations of 2 variables out of M)
    minimumL = M * (M - 1) / 2

    # if L is less than M C 2, no solution exists
    if L < minimumL:
        print (countNodes_fc)
        return -1, []

    # initialize the domains of all the variables
    for i in range (int(minimumL), L + 1):
        initialDomains = []
        for j in range(0, i + 1):
            initialDomains.append(j)

        # Initialise the dictionary containing variables and their corresponding domains
        domains = {}
        for j in range(0, M):
            domains[j + 1] = list(initialDomains)

        # Call recursiveBT with positions, domains, mCurrent, distances and Length as parameters
        newL, newPos = recursiveFC(positions, domains, 1, [], i)

        # return L + 1 and the positions returned by recursiveBT
        if newL != -1:
            print (newPos)
            return newL, newPos
    print (countNodes_fc)
    return -1, []

def recursiveFC(positions, domains, mCurrent, distancesCurrent, L):
    # Input parameters:
        # positions: contains the positions which have been assigned to the markers (-1 indicates not yet assigned)
        # domains: contains the domain of values that every marker holds
        # mCurrent: is the current marker which has to be assigned a position
        # distancesCurrent: contains the distances between the markers which have already been assigned positions
        # L: length of the Golomb ruler
    # Returns:
        # first length at which the solution is found
        # positions of the markers

    # Increment the count of the nodes when this function is called
    global countNodes_fc
    countNodes_fc = countNodes_fc + 1

    # If -1 is not present in the positions of the markers, then positions have been assigned to all markers.
    if -1 not in positions.values():
        return [positions[mCurrent - 1], positions]

    # If any of the domains of the variables is empty then no solution exists
    # for the assignment that was passed to this function (Forward checking)
    for i in domains.keys():
        if len(domains[i]) == 0:
            return [-1, positions]

    # Iterate over all the positions in the domain of mCurrent and check if there is a solution
    for mPos in domains[mCurrent]:
        allocateFlag = True

        # Create a new distances list which contains the distances between the markers which have already been assigned positions
        distancesNew = list(distancesCurrent)
        distancesTemp = []

        # Calculate the distance between mPos and positions of all the assigned markers.
        for i in range(1, mCurrent):
            distance = abs(mPos - positions[i])

            # If the calculated distance is already present in the new distance list set allocateFlag to False and consider next mPos from the
            # domain of the mCurrent for allocation
            if distance in distancesNew:
                allocateFlag = False
                break

            # If the calculated distance is ndistancesCurrent
            distancesTemp.append(distance)
        if not allocateFlag:
            continue

        # Create a new positions dictionary to store the position of the markers so that the original positions dictionary can be used for backtracking purposes
        positionsNew = dict(positions)
        positionsNew[mCurrent] = mPos

        # Create a new domains dictionary to store the domains of the markers so that original domains dictionary can be used for backtracking purposes
        domainsNew = {}
        for elem in domains.keys():
            domainsNew[elem] = list(domains[elem])

        # Removing mPos from the domains of all future markers
        for i in range(mCurrent + 1, len(positions.keys()) + 1):
            if mPos in domainsNew[i]:
                domainsNew[i].remove(mPos)

        # pruning domains of future markers according to constraints
            # 1. remove the positions from the domains of all markers where positions = allocated positions + distances iff positions <= L
            # 2. remove the positions from the domains of all markers where positions = allocated positions - distances iff positions >= 0
        for item in positionsNew.values():
            if item  == -1:
                break
            for elem in distancesTemp:
                positionToRemove1 = item + elem
                if positionToRemove1 <= L:
                    for i in range(mCurrent + 1, len(positions.keys()) + 1):
                        if positionToRemove1 in domainsNew[i]:
                            domainsNew[i].remove(positionToRemove1)

                positionToRemove2 = item - elem
                if positionToRemove2 >= 0:
                    for i in range(mCurrent + 1, len(positions.keys()) + 1):
                        if positionToRemove2 in domainsNew[i]:
                            domainsNew[i].remove(positionToRemove2)

        # if any of the domains are empty after pruning, set end_flag to True and then Backtrack
        # else call recursiveFC with positionsNew, domainsNew, mCurrent + 1, distancesNew and L as parameters
        end_flag = False
        for i in domainsNew.keys():
            if len(domainsNew[i]) == 0:
                end_flag = True
        if not end_flag:
            retL, retPositions = recursiveFC(positionsNew, domainsNew, mCurrent + 1, distancesNew, L)
            # if retL is not -1 then solution exists, return retL, and ret Positions
            if retL != -1:
                return retL, retPositions
    # all mPos are tried and the solution doesn't exits, return -1 and positions
    return -1, positions