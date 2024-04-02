import numpy as np

def ransac(x, fittingfn, distfn, degenfn, s, t, feedback=0, maxDataTrials=100, maxTrials=1000, p=0.99):
    bestM = None      # Sentinel value allowing detection of solution failure.
    bestscore =  0
    trialcount = 0
    N = 1            # Dummy initialisation for number of trials.
    
    while N > trialcount:
        # Select at random s datapoints to form a trial model, M.
        # In selecting these points we have to check that they are not in
        # a degenerate configuration.
        degenerate = True
        count = 1
        while degenerate:
            # Generate s random indices in the range 0 to npts-1
            ind = np.random.choice(x.shape[1], s, replace=False)
            
            # Test that these points are not a degenerate configuration.
            degenerate = degenfn(x[:, ind])
            
            if not degenerate:
                # Fit model to this random selection of data points.
                # Note that M may represent a set of models that fit the data,
                # in this case M will be a list of models
                M = fittingfn(x[:, ind])
                
                # Check if fittingfn returned any models
                if not M:
                    degenerate = True
                
            # Safeguard against being stuck in this loop forever
            count += 1
            if count > maxDataTrials:
                if feedback:
                    print('Unable to select a nondegenerate data set')
                break
        
        if not degenerate:
            # Evaluate distances between points and model returning the indices
            # of elements in x that are inliers. Additionally, if M is a list
            # of possible models, distfn will return the model that has the most
            # inliers. After this call, M will be a single object representing
            # the best model.
            inliers, M = distfn(M, x, t)
        else:
            inliers = []

        # Find the number of inliers to this model.
        ninliers = len(inliers)
        
        if ninliers > bestscore:
            # Largest set of inliers so far, update records
            bestscore = ninliers
            bestinliers = inliers
            bestM = M
            
            # Update estimate of N, the number of trials to ensure we pick,
            # with probability p, a data set with no outliers.
            fracinliers = ninliers / x.shape[1]
            pNoOutliers = 1 - fracinliers**s
            pNoOutliers = max(np.finfo(float).eps, pNoOutliers)  # Avoid division by -Inf
            pNoOutliers = min(1 - np.finfo(float).eps, pNoOutliers)  # Avoid division by 0.
            N = np.log(1 - p) / np.log(pNoOutliers)

        trialcount += 1
        if feedback:
            print(f'trial {trialcount} out of {np.ceil(N)}', end='\r')

        # Safeguard against being stuck in this loop forever
        if trialcount > maxTrials:
            print(f'ransac reached the maximum number of {maxTrials} trials')
            break

    if feedback:
        print('\n')

    if bestM is not None:   # We got a solution
        M = bestM
        inliers = bestinliers
    else:
        M = None
        inliers = []
        print('ransac was unable to find a useful solution')

    return M, inliers

