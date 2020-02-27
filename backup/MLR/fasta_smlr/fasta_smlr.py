# coding=utf-8
import logging

import numpy as np
import time

logger = logging.getLogger("fasta-smlr")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def w_norm(w):
    """ w is a matrix, perform a element-wise l2-norm """
    return np.sqrt(np.sum(w * w))

def safe_log(x, minval=1e-10):
    return np.log(x.clip(min=minval))

def softmax(x):
    x -= x.max(axis=1).reshape([-1, 1])  # Very important, avoid NAN of EXP function
    p = np.e ** x
    p /= p.sum(axis=1).reshape([-1, 1])
    return p

def obj(X, y, theta):
    """ objective of the differentiable function g """
    m = X.shape[0]
    margin = X.dot(theta)
    p = softmax(margin)
    loss_obj = -safe_log(p[np.arange(len(p)), y]).sum() / m
    return loss_obj

def gradient(X, y, theta):
    """ gradient of the differentiable function g """
    m = X.shape[0]
    margin = X.dot(theta)
    p = softmax(margin)
    p[np.arange(len(p)), y] -= 1
    dW = X.T.dot(p) / m
    return dW

def shrinkage(a, kappa):
    a_ = np.sign(a)*np.maximum(np.abs(a)-kappa, 0)
    return a_

def fasta_smlr(A, At, b, lmd, x0, opts):
    # Define ingredients for FASTA
    f    = lambda theta: obj(A, b, theta)
    g    = lambda theta: lmd*np.sum(np.abs(theta))
    grad = lambda theta: gradient(A, b, theta)
    prox = lambda a, t: shrinkage(a, lmd*t)

    return fasta(A, At, f, grad, g, prox, x0, opts)

def fasta(A, At, f, gradf, g, proxg, x0, opts):

    # Set default opts
    setDefault(opts)

    if opts['verbose']:
        logger.info('{}FASTA:\n\t\tmode = {}\n\t\tmaxIters = {},\n\t\ttol = {}\n\t\ttau = {}'.format(
            opts['stringHeader'], opts['mode'], opts['maxIters'], opts['tol'], opts['tau'])
        )

    # Record some frequently used information from opts
    tau1      = opts['tau']                     # initial stepsize
    max_iters = opts['maxIters']                # maximum iterations before automatic termination
    W         = opts['window']                  # lookback window for non-montone line search

    # Allocate memory
    iterates   = [x0]
    residual   = np.zeros((max_iters, 1))       # Residuals
    normalizedResid = np.zeros((max_iters, 1))  # Normalized residuals
    taus       = np.zeros((max_iters, 1))       # Stepsizes
    fVals      = np.zeros((max_iters, 1))       # The value of 'f', the smooth objective term
    objective  = np.zeros((max_iters + 1, 1))   # The value of the objective function (f+g)
    totalBacktracks = 0                         # How many times was backtracking activated?
    backtrackCount  = 0                         # Backtracks on this iterations

    # Intialize array values
    x1       = x0
    f1       = f(x0)
    fVals[0] = f1
    gradf1   = gradf(x0)

    # To handle non-monotonicity
    maxResidual       = -np.inf # Stores the maximum value of the residual that has been seen. Used to evaluate stopping conditions.
    minObjectiveValue = np.inf  # Stores the best objective value that has been seen.  Used to return best iterate, rather than last iterate

    # Initialize additional storage required for FISTA
    if opts['accelerate']:
        x_accel1 = x0
        alpha1 = 1

    # If user has chosen to record objective, then record initial value
    if opts['recordObjective']: # record function values
        objective[0] = f1 + g(x0)

    # Begin Loop
    start = time.time()
    for i in range(max_iters):
        print "iter: {}".format(i)
        # Rename iterates relative to loop index.  "0" denotes index i, and "1" denotes index i+1
        x0 = x1         # x_i <- x_{i+1}
        gradf0 = gradf1 # gradf0 is now \nabla f(x_i)
        tau0 = tau1     # \tau_i <- \tau_{i+1}

        # FBS step: obtain x_{i+1} from x_i
        x1hat = x0 - tau0 * gradf0  # Define \hat x_{i+1}
        x1 = proxg(x1hat, tau0)     # Define x_{i+1}

        # Non-monotone backtracking line search
        Dx = x1 - x0
        f1 = f(x1)
        if opts['backtrack']:
            M = max( fVals[max(i-W,0): max(i-1,1)] )
            backtrackCount = 0
            # Note: 1e-12 is to quench rounding errors
            while f1-1e-12 > M + np.sum(Dx * gradf0) + w_norm(Dx)**2/(2*tau0) and backtrackCount < 20:
                # TODO:
                # logger.info("##### backtracking ")
                tau0 = tau0 * opts['stepsizeShrink'] # shrink stepsize
                x1hat = x0 - tau0 * gradf0           # redo the FBS
                x1 = proxg(x1hat, tau0)
                f1 = f(x1)
                Dx = x1 - x0
                backtrackCount += 1
            totalBacktracks = totalBacktracks + backtrackCount

        if opts['verbose'] and backtrackCount > 10:
            logger.info('{}\tWARNING: excessive backtracking ({} steps), current stepsize is {}\n'.format(
                opts['stringHeader'], backtrackCount, tau0)
            )

        # Record convergence information
        taus[i] = tau0 # stepsize
        residual[i] = w_norm(Dx) / tau0 # Estimate of the gradient, should be zero at solution
        maxResidual = max(maxResidual, residual[i])
        normalizer  = max(w_norm(gradf0), w_norm(x1-x1hat)/tau0) + opts['eps_n']
        normalizedResid[i] = residual[i] / normalizer
        fVals[i] = f1
        if opts['recordObjective']: # Record function values
            objective[i+1] = f1 + g(x1)
            newObjectiveValue = objective[i+1]
        else:
            newObjectiveValue = residual[i] # Use the residual to evalue quality of iterate if we don't have objective

        if opts['recordIterates']:
            iterates.append(x1)
        # TODO:
        # logger.info("iter: {}, newObjectiveValue: {}, tau: {}".format(i, newObjectiveValue, tau0))

        if newObjectiveValue < minObjectiveValue: # Methods is non-monotone:  Make sure to record best solution
            bestObjectiveIterate = x1
            minObjectiveValue = newObjectiveValue

        if opts['verbose'] > 1:
            objStr = ', objective = {}\n'.format(objective[i+1]) if opts['recordObjective'] else '\n'
            logger.info('{}{}: resid = {}, backtrack = {}, tau = {} {}'.format(
                opts['stringHeader'], i, residual[i], backtrackCount, tau0, objStr
            ),)

        # Test stopping criteria
        if opts['stopNow'](x1,i,residual[i],normalizedResid[i],maxResidual,opts) or i>=max_iters-1:
            outs = {}
            outs['solveTime'] = time.time()
            outs['runningTime'] = outs['solveTime'] - start
            outs['residuals'] = residual[:i]
            outs['stepsizes'] = taus[:i]
            outs['normalizedResiduals'] = normalizedResid[:i]
            outs['objective'] = objective[:i]
            outs['backtracks'] = totalBacktracks
            outs['L'] = opts['L']
            outs['initialStepsize'] = opts['tau']
            outs['iterationCount'] = i
            if not opts['recordObjective']: outs['objective'] = 'Not Recorded'
            if opts['recordIterates']: outs['iterates'] = iterates
            sol = bestObjectiveIterate
            if opts['verbose']:
                logger.warn('{}\tDone:  time = {} secs, iterations = {}\n'.format(
                    opts['stringHeader'], (outs['solveTime']-start), outs['iterationCount']
                ))
            return sol, outs

        if opts['adaptive'] and not opts['accelerate']:
            # Compute stepsize needed for next iteration using BB/spectral method
            gradf1 = gradf(x1)
            Dg = gradf1 + (x1hat - x0) / tau0 # Delta_g, note that Delta_x was recorded above during backtracking
            dotprod = np.real(np.sum(Dx * Dg))
            tau_s = w_norm(Dx)**2 / dotprod     # First BB stepsize rule
            tau_m = dotprod / w_norm(Dg)**2     # Alternate BB stepsize rule
            tau_m = max(tau_m, 0)
            if 2 * tau_m > tau_s: # Use "Adaptive"  combination of tau_s and tau_m
                tau1 = tau_m
            else:
                tau1 = tau_s - .5 * tau_m     # Experiment with this param
            if tau1 <=0 or np.isinf(tau1) or np.isnan(tau1): # Make sure step is non-negative
                tau1 = tau0 * 1.5 # let tau grow, backtracking will kick in if stepsize is too big

        # TODO: 该改这里了
        if opts['accelerate']:
            x_accel0 = x_accel1
            d_accel0 = d_accel1
            alpha0 = alpha1
            x_accel1 = x1
            d_accel1 = d1
            if opts['restart'] and np.dot((x0 - x1)[:,0], (x1-x_accel0)[:,0])>0:
                alpha0 = 1
            # Calculate acceleration parameter
            alpha1 = (1 + np.sqrt(1 + 4 * alpha0**2)) / 2
            print "alpha: {}".format(alpha1)
            # Over-relax/predict
            x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)
            # Compute the gradient needed on the next iteration
            gradf1 = At_func(gradf(d1))
            fVals[i] = f(d1)
            tau1 = tau0

        if not opts['adaptive'] and not opts['accelerate']:
            gradf1 = gradf(x1)
            tau1 = tau0

def setDefault(opts):
    """ Fill in the struct of options with the default values """
    # maxIters: The maximum number of iterations
    if not opts.has_key('maxIters'):
        opts['maxIters'] = 1000
    # tol:  The relative decrease in the residuals before the method stops
    if not opts.has_key('tol'):
        opts['tol'] = 1e-3
    # verbose:  If 'true' then print status information on every iteration
    if not opts.has_key('verbose'):
        opts['verbose'] = False
    # recordObjective:  If 'true' then evaluate objective at every iteration
    if not opts.has_key('recordObjective'):
        opts['recordObjective'] = False
    # recordIterates:  If 'true' then record iterates in cell array
    if not opts.has_key('recordIterates'):
        opts['recordIterates'] = False
    # adaptive:  If 'true' then use adaptive method.
    if not opts.has_key('adaptive'):
        opts['adaptive'] = True
    # accelerate:  If 'true' then use FISTA-type adaptive method.
    if not opts.has_key('accelerate'):
        opts['accelerate'] = False
    # restart:  If 'true' then restart the acceleration of FISTA.
    # This only has an effect when opts.accelerate=true
    if not opts.has_key('restart'):
        opts['restart'] = True
    # backtrack:  If 'true' then use backtracking line search
    if not opts.has_key('backtrack'):
        opts['backtrack'] = True
    # stepsizeShrink:  Coefficient used to shrink stepsize when backtracking kicks in
    if not opts.has_key('stepsizeShrink'):
        opts['stepsizeShrink'] = 0.2     # The adaptive method can expand the stepsize, so we choose an aggressive value here
        if not opts['adaptive'] or opts['accelerate']:
            opts['stepsizeShrink'] = 0.5 # If the stepsize is monotonically decreasing, we don't want to make it smaller than we need

    # Create a mode string that describes which variant of the method is used
    opts['mode'] = 'plain'
    if opts['adaptive']:
        opts['mode'] = 'adaptive'
    if opts['accelerate']:
        if opts['restart']:
            opts['mode'] = 'accelerated(FISTA)+restart'
        else:
            opts['mode'] = 'accelerated(FISTA)'

    # W:  The window to look back when evaluating the max for the line search
    if not opts.has_key('window'):
        opts['window'] = 10
    # eps_r:  Epsilon to prevent ratio residual from dividing by zero
    if not opts.has_key('eps_r'):
        opts['eps_r'] = 1e-8
    # eps_n:  Epsilon to prevent normalized residual from dividing by zero
    if not opts.has_key('eps_n'):
        opts['eps_n'] = 1e-8

    # L:  Lipschitz constant for smooth term.  Only needed if tau has not been
    # set, in which case we need to approximate L so that tau can be
    # computed.
    if (not opts.has_key('L') or opts['L']<=0) and (not opts.has_key('tau') or opts['tau']<=0):
        opts['tau'] = 0.8
    assert opts['tau']>0, "Invalid step size: {}".format(opts['opts'])

    # Set tau if L was set by user
    if not opts.has_key('tau') or opts['tau']<=0:
        opts['tau'] = 1.0/opts['L']
    else:
        opts['L'] = 1/opts['tau']

    if not opts.has_key('stringHeader'): # This functions gets evaluated on each iterations, and results are stored
        opts['stringHeader'] = ''

    # The code below is for stopping rules
    # The field 'stopNow' is a function that returns 'true' if the iteration
    #  should be terminated.  The field 'stopRule' is a string that allows the
    #  user to easily choose default values for 'stopNow'.  The default
    #  stopping rule terminates when the relative residual gets small.
    if opts.has_key('stopNow'):
        opts['stopRule'] = 'custom'
    if not opts.has_key('stopRule'):
        opts['stopRule'] = 'hybridResidual'
    if opts['stopRule'] == 'residual':
        opts['stopNow'] = lambda x1,iter,resid,normResid,maxResidual,opts: resid < opts['tol']
    if opts['stopRule'] == 'iteration':
        opts['stopNow'] =  lambda x1,iter,resid,normResid,maxResidual,opts: iter > opts['maxIters']
    if opts['stopRule'] == 'normalizedResidual':
        opts['stopNow'] = lambda x1,iter,resid,normResid,maxResidual,opts: normResid < opts['tol']
    if opts['stopRule'] == 'ratioResidual':
        opts['stopNow'] = lambda x1,iter,resid,normResid,maxResidual,opts: resid/(maxResidual+opts['eps_r']) < opts['tol']
    if opts['stopRule'] == 'hybridResidual':
        opts['stopNow'] = lambda x1,iter,resid,normResid,maxResidual,opts: (
            resid / (maxResidual + opts['eps_r']) < opts['tol'] or normResid < opts['tol']
        )

