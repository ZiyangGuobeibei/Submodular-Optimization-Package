# submodular_algs.jl
# Chris Harshaw & Ziyang Guo, Yale University
# August 2019
#
# These are several submodular function maximization algorithms, including the discrete greedy
# algorithm and faster variants of the continuous greedy algorithm & rounding.
#

# TODO
# 1. Add an option to continuous greedy to track # of function evaluations

using DataStructures # for priority queue
using StatsBase # for sampling
using LinearAlgebra # matrix operations (we haven't used this yet)
using Printf # this is for printing (which we haven't used yet)
using Revise # this is for for jupyter notebooks

function greedy_alg(f_diff, n, k; lazy=false, lazier=false, epsilon=0.05)
    """
    # greedy_alg
    # The greedy algorithm for maximizing a monotone submodualr set function
    # subject to a cardinality constraint. Options for faster variants, like
    # "lazy greedy" which uses lazy evaluations of marginal gains and
    # "lazier than lazy greedy" which subsamples the ground set at each iteration.
    #
    # Input
    # 	f_diff		a function to compute marginal differences, f(Int64 e , Set{Int64} S) = f( S + e) - f(S)
    #   n           size of the ground set
    #   k           the cardinality constraint
    #   lazy        option to use lazy evaluations
    #   lazier      option to sample the ground set
    #   epslion     parameter in lazier than lazy greedy algorithm
    #
    # Output
    #   S           the set produced by greedy algorithm
    """

    # initialize the set
    S = Set{Int64}()

    # set the sample size to n/k log(1/eps)
    sample_size = min(convert(Int64, ceil(n/k * log(1/epsilon))), n)

    # initialize data structure for lazy evaluations
    if lazy
        if lazier
            # create a marginal gains array (new pq is instantianted each iteration)
            recent_gains = Inf * ones(n)
        else

            # create a dedicated priority queue, initialize
            pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)
            for i=1:n
                enqueue!(pq, i, f_diff(i,S))
            end
        end
    end

    # greedily select best element, k iterations; we do one of 4 options
    for i=1:k

        # 1. Simple Lnear Search - not lazy, not lazier
        if !lazy & !lazier

            # initialize
            max_gain = -Inf
            max_elm = -1

            # linear search
            for j in 1:n
                gain = f_diff(j,S)
                if gain > max_gain
                    max_gain = gain
                    max_elm = j
                end
            end

        # 2. Priority Queue - lazy, not lazier
        elseif lazy & !lazier

            while true
                # get the top element & compute its marginal gain
                top_elm = dequeue!(pq)
                top_gain = f_diff(top_elm, S)

                # if it's bigger than nexr top marginal gain, it's max
                next_elm, next_gain = peek(pq)
                if top_gain >= next_gain
                    max_elm = top_elm
                    max_gain = top_gain
                    break
                else
                    enqueue!(pq, top_elm, top_gain)
                end
            end


        # 3. Subsampled Ground Set - lazier, not lazy
        elseif !lazy & lazier

            # subsample ground set to get reduced search space
            search_set = sample(collect(1:n), sample_size, replace=false)

            # initialize
            max_gain = -Inf
            max_elm = -1

            # linear search on subsample
            for j in search_set
                gain = f_diff(j,S)
                if gain > max_gain
                    max_gain = gain
                    max_elm = j
                end
            end

        # 4. Prioerty Queue AND Subsampled Ground Set
        else

            # subsample ground set to get reduced search space
            search_set = sample(collect(1:n), sample_size, replace=false)

            # create a priority queue on the subsampled ground set
            pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)
            for i in search_set
                enqueue!(pq, i, recent_gains[i])
            end

            # go thru priority queue
            while true

                # get the top element & compute its marginal gain
                top_elm = dequeue!(pq)
                top_gain = f_diff(top_elm, S)
                recent_gains[top_elm] = top_gain # update array of recent marginal gains

                # if it's bigger than nexr top marginal gain, it's max
                next_elm, next_gain = peek(pq)
                if top_gain >= next_gain
                    max_elm = top_elm
                    max_gain = top_gain
                    break
                else
                    enqueue!(pq, top_elm, top_gain)
                end
            end
        end

        # udpate the solution set S
        if max_gain > 0
            push!(S,max_elm)
        else
            break
        end
    end

    return S
end


function set_to_vec(S, n)
    """
    # set_to_vec
    # Convert a set to its 0/1 indicator vector
    #
    # Input
    #   S   a set
    #   n   size of ground set
    #
    # Output
    #   z   the 0/1 indicator
    """
    z = [i in S ? 1 : 0 for i=1:n]
    return z
end


function mle_est_val(f, x, batch_size)
    """
    # exact_fw
    # A sampling algorithm which estimates the multilinear extension of f
    # at x, i.e. F(x)
    #
    # Input
    # 	f		    a value oracle for the submodular function
    #   x 	        a vector in [0,1]^n
    #   batch_size  number of samples to use in estimate
    #
    # Output
    #   val 	estimate of f(x)
    """

    val = 0
    for i=1:batch_size

        # get a random set S
        S = Set{Int64}()
        for j=1:n
            if rand() <= x[j] # include j with probability x(j)
                push!(S, j)
            end
        end

        # approximate the value f(S)
        val += f(S)
    end
    val /= batch_size # normalize
    return val
end


function mle_stochastic_grad_est(f_diff, x)
	"""
    # exact_fw
    # A sampling algorithm which estimates the gradinet of the multilinear
    # extension.
    #
    # Input
    # 	f_diff		a function to compute marginal differences, f(Int64 e , Set{Int64} S) = f( S + e) - f(S)
    #   x 			a vector in [0,1]^n
    #
    # Output
    #   grad 		stochastic gradient estimate
    """

    n = length(x)
    grad = zeros(n)

    # compute an estimate in each coordinate
    for i=1:n

        # get a random set NOT containing i
        S = Set{Int64}()
        for j=1:n
            if i == j # do not include i
                continue
            end
            if rand() <= x[j] # include j with probability x(j)
                push!(S, j)
            end
        end

        # approximate the gradient coordiante
        grad[i] = f_diff(i, S)
    end

    return grad
end

function mle_grad_at_zero(f_diff, n)
	"""
    # mle_grad_at_zero
    # Obtains the exact gradient at zero. No large sampling required.
    #
    # Input
    # 	f_diff		a function to compute marginal differences, f(Int64 e , Set{Int64} S) = f( S + e) - f(S)
    #   n			number of elements in the ground set
    #
    # Output
    #   grad 		gradient at zero
    """

    S = Set{Int64}()
    grad = [f_diff(i, S) for i=1:n]

    return grad
end

function hessian_prod(f_diff, x, S, batch_size)
    """
    # hessian_prod
    # Approximates the product of the hessian of mle at x and the 0/1 vector 1_S
    #
    # Input
    # 	f_diff		a function to compute marginal differences, f(Int64 e , Set{Int64} S) = f( S + e) - f(S)
    #   x           the current iterate
    #   S           a set containing non-zero indices
    #   batch_size  the batch size, number of samples to use
    #
    # Output
    #   prod_approx the approximation to the matrix vector product
    """

    # get dimensions
    n = length(x)

    prod_approx = zeros(n)
    for b=1:batch_size
        for i=1:n
            for e in S

                # approximate H(x(a))_{i,e} = \frac{ partial^2 }{ \partial x_i x_e} F(x(a))

                # get a random set containing NEITHER i nor e
                S_rand = Set{Int64}()
                for j=1:n
                    if (j == i) | (j == e) # do not include i
                        continue
                    end
                    if rand() <= x[j] # include j with probability x(j)
                        push!(S_rand, j)
                    end
                end

                # approximate hessian by f( i | S + e) - f( e | S)
                prod_approx[i] += f_diff(i, union(S_rand, e)) - f_diff(e, S_rand)
            end
        end
    end
    prod_approx /= batch_size # normalization
    return prod_approx
end

function continuous_greedy(f_diff, n, T, batch_size, mat_lin_opt, merge_base; fw_type="average", measured=false, verbose=true)
	"""
    # continuous_greedy
    # A few stochastic variants of the Frank-Wolfe algorithm for maximizing multilinear relaxation.
    # Supports continuous greedy for monotone submodular maximization and measured continuous greedy too.
    #
    # Input
    # 	f_diff		a function to compute marginal differences, f(Int64 e , Set{Int64} S) = f( S + e) - f(S)
    #   n           number of elements in ground set
    #   T           number of iterations to run continuous greedy
    #   batch_size	the number of samples to use in the stochastic gradient oracles at each iteration
    #   mat_lin_opt an oracle for linear optimization over the matroid
    #   fw_type     the stochastic fw algorithm used. Either "average", "momentum", or "spider"
    #   measured    set true to run measured continuous greedy, default is false.
    #   verbose     option true/false for printing algorithm state
    #
    # Output
    #   x           the last iterate returned by FW
    #   v_set       matroid bases chosen by FW (length T array of sets)
    """

    # initialize x = 0 and v_set
    x_prev = zeros(n)
    x = zeros(n)
    proxy_grad = zeros(n)
    v_set = Array{Set{Int64}, 1}() # TODO: make this be a fixed length T?

    # run T FW iterations
    for t=1:T

        # get previous x
        x_prev = copy(x)

        # at the first iteration, proxy gradient = exact gradient
        if t ==1
            proxy_grad = mle_grad_at_zero(f_diff, n)
        else

            if fw_type == "average"

                # estimate gradient by sampling
                batch_grad = zeros(n)
                for i=1:batch_size
                    batch_grad += mle_stochastic_grad_est(f_diff, x)
                end
                proxy_grad = batch_grad / batch_size

            elseif fw_type == "momentum"

                # estimate the gradient by sampling
                batch_grad = zeros(n)
                for i=1:batch_size
                    batch_grad += mle_stochastic_grad_est(f_diff, x)
                end
                batch_grad /= batch_size

                # update the proxy gradient with momentum
                rho = 2 /(t + 2)^(2/3)
                proxy_grad = (1 - rho)*proxy_grad + rho * batch_grad

            elseif fw_type == "spider"

                # choose a point uniformly between x_t and x_{t-1}
                a = rand()
                x_a = x_prev + (a/T) * v

                # proxy gradient is a Hessian - vector product
                proxy_grad = hessian_prod(f_diff, x_a, S, batch_size) / T

            else
                error("Frank Wolfe type not recognized")
            end

        end

        # perform linear optimization
        if measured
            S = mat_lin_opt( (ones(n) - x) .* proxy_grad ) # measured LMO
        else
            S = mat_lin_opt(proxy_grad) # normal LMO
        end

        # update vector x
        v = v = set_to_vec(S,n)
        if measured
            x += (v / T) .* (ones(n) - x) # measured update
        else
            x += v / T # normal update
        end

        # update sets obtained by LMO
        push!(v_set, S)

        # this does the merging step at the same time
        # # merge bases
        # if t == 1
        # 	round_x = S
        # else
        #     round_x = merge_base(round_x, S, (t-1)/T, 1/T)
        # end

        # print algorithm info if verbose
        if verbose
            @printf("\nIteration %d of %d\n", t, T)
            println("\tx was ", x_prev)
            println("\tproxy grad was ", proxy_grad)
            println("\tfw linear subroutine produced set ", S)
            println("\tCorresponding vector v was ", v)
            # println("\tNewly rounded x is ", round_x)
        end
    end

    return x, v_set
end

function swap_rounding(base_list, merge_base; weights=nothing)
    """
    # swap_rounding
    # Swap rounding technique for matroid bases.
    #
    # Input
    # 	base_list   an array of bases (sets) of the matroid
    # 	merge_base 	an oracle for merging two bases of the matroid
    #   weights     (optional) a liist of weights for the bases; default is uniform weight
    #
    # Output
    #   round_x     base of the matroid which is rounded (set)
    """

    # get number of bases
    T = length(base_list)

    # fix weights to be uniform if not given
    if weights == nothing
        weights = ones(T)
    end

    # merge bases one at a time
    curr_weight = weights[1]
    round_x = base_list[1]
    for t = 2:T
        round_x = merge_base(round_x, base_list[t], curr_weight, weights[t])
        curr_weight += weights[t]
    end

    return round_x
end


function knapsack_greedy(f_diff, c, b, n; lazy = false)
    """
    #Knapsack greedy algorithm:
    #run plain greedy with the knapsack constraint

    # Input
    # 	f_diff		a function to compute the function value on sets, f_diff(Int64 e) = f(S union e) - f(S)
    #   c           a vector recording the cost of all elements
    #   b           total budget
    #   n           size of the ground set
    #   lazy        option to use lazy evaluations
    #
    # Output
    #   S           the set produced by greedy algorithm
    #   value       the function value at the output set
    #
    # TODO
    # 1. Add in the lazy features
    # 2. Test in Jupyter Notebook using Chris's functions
    """

    #First initialize the set that records the selections so far and the current available elements
    if !lazy
        A = Set{Int64}(1 : n)
        budget = b
        value = 0
        S = Set{Int64}()

        #run the algorithm until A becomes empty
        while !isempty(A)

            #Initialize the variables:
            cur = Set{Int64}()
            max_so_far = -Inf
            max_index = 0

            #Try to search for the "best" element
            for i in A

                #Check if the budget even allows for the current element to be added
                if c[i] > budget
                    push!(cur, i)
                    continue
                end

                #Check if the current marginal utility is larger than the best so far
                cur_margin = f_diff(i, S)
                if cur_margin > max_so_far
                    max_so_far = cur_margin
                    max_index = i
                end
            end

            #Check if the "best" element offers any marginal utility
            if max_so_far > 0

                #Remove the useless elements, update the cost, and push the element into the set
                push!(S, max_index)
                setdiff!(A, push!(cur, max_index))
                budget = budget - c[max_index]
                value = value + max_so_far
            else
                break
            end
        end
    else
        #Initialize the priority queue and the return values
        pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)
        value = 0
        budget = b
        S = Set{Int64}()

        #Initialize all the key value pairs
        for i in 1 : n
            enqueue!(pq, i, f_diff(i, S))
        end
            #Pick the best elemnt from the priority queue until the updated difference is the largest
        while !isempty(pq)
            #Pick out the best element so far
            cur_index, cur_margin = peek(pq)
            dequeue!(pq)

            #Check if the priority queue is empty now
            if cur_margin <= 0
                return S, value
            end

            #Delete the element if the updated marginal utility is non-positive or if the budget is not enough
            cur_margin = f_diff(cur_index, S)
            if cur_margin <= 0 || c[cur_index] > budget
                continue
            elseif isempty(pq) || cur_margin >= peek(pq)[2]
                push!(S, cur_index)
                budget = budget - c[cur_index]
                value = value + cur_margin
            else
                enqueue!(pq, cur_index, cur_margin)
            end
        end
    end

    return S, value
end

function knapsack_ratio_greedy(f_diff, c, b, n; lazy = false)
    """
    #Knapsack ratio greedy algorithm:
    #run ratio greedy with the knapsack constraint

    # Input
    # 	f_diff		a function to compute the function value on sets, f_diff(Int64 e) = f(S union e) - f(S)
    #   c           a vector recording the cost of all elements
    #   b           total budget
    #   n           size of the ground set
    #   lazy        option to use lazy evaluations
    #
    # Output
    #   S           the set produced by ratio greedy algorithm
    #   value       the function value at the output set
    #
    # TODO
    # 1. Add in the lazy features
    # 2. Test in Jupyter Notebook using Chris's functions
    """
    if !lazy
    #Initialize the variables
        A = Set{Int64}(1 : n)
        budget = b
        value = 0
        S = Set{Int64}()

        while !isempty(A)

            #Initialize the variables in the current run
            cur = Set{Int64}()
            max_so_far = -Inf
            max_ratio_so_far = -Inf
            max_index = 0

            #Visit every element that are still in the set
            for i in A
                #Take the unaffordable element out
                if c[i] > budget
                    push!(cur, i)
                    continue
                end

                #Otherwise compute the marginal utility and compare with the best so far
                cur_margin = f_diff(i, S)
                cur_ratio = cur_margin / c[i]
                if cur_ratio > max_ratio_so_far
                    max_ratio_so_far = cur_ratio
                    max_so_far = cur_margin
                    max_index = i
                end
            end

            #See if the best marginal utility so far is positive
            if max_so_far > 0
                push!(S, max_index)
                setdiff!(A, push!(cur, max_index))
                budget = budget - c[max_index]
                value = value + max_so_far
            else
                break
            end
        end
    else
        #Initialize the return values and the priority queue
        pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)
        value = 0
        budget = b
        S = Set{Int64}()

        #Initialize the key value pairs in pq
        for i in 1 : n
            enqueue!(pq, i, f_diff(i, S) / c[i])
        end

        #Pick elements until pq is empty
        while !isempty(pq)
            #Pick out the best element so far
            cur_index, cur_margin = peek(pq)
            dequeue!(pq)

            if cur_margin <= 0
                return S, value
            end

            #Delete the element if the updated marginal utility is non-positive or if the budget is not enough
            cur_margin = f_diff(cur_index, S) / c[cur_index]
            if cur_margin <= 0 || c[cur_index] > budget
                continue
            elseif isempty(pq) || cur_margin >= peek(pq)[2]
                push!(S, cur_index)
                budget = budget - c[cur_index]
                value = value + cur_margin
            else
                enqueue!(pq, cur_index, cur_margin)
            end
        end
    end

    return S, value
end

function knapsack_greedy_final(f_diff, c, b, n; lazy = false)
    """
    #Knapsack greedy algorithm:
    #first run plain greedy on the set
    #then run the greedy on the benefit-cost ratios
    #take the maximum of the output of these two algorithms

    # Input
    # 	f_diff		a function to compute the marginal utility of an element e given a set S
    #   c           a vector recording the cost of all elements
    #   b           total budget
    #   n           size of the ground set
    #   lazy        option to use lazy evaluations
    #
    # Output
    #   S           the set produced by greedy algorithm
    """

    S1, value1 = knapsack_greedy(f_diff, c, b, n, lazy = lazy)
    S2, value2 = knapsack_ratio_greedy(f_diff, c, b, n, lazy = lazy)

    if value1 > value2
        return S1
    else
        return S2
    end
end

function double_greedy(f_diff, n; random = false)
    """
    #Deterministic Version:
    #double greedy algorithm:
    #Start with X = emptyset, Y = Set(1 : n)
    #Iterate from 1 to n and, for each element, greedily determine whether to adjoin the element to X or take the element out from Y
    #Return the set in the end

    #Randomized Version:
    #Proceed as in the deterministic version
    #Add in some "smoothing" when deciding whether to adjoin the element to X or to remove it from Y

    # Input
    # 	f_diff		a function to compute the marginal utility of an element e given a set S
    #   n           size of the ground set
    #   rand        option to use the randomized version
    #
    # Output
    #   X (or Y)    the set produced by greedy algorithm
    """

    #Initialize X and Y
    X = Set{Int64}()
    Y = Set{Int64}(1 : n)

    #Iterate through all elements from 1 to n
    for i in 1 : n
        #Calculate f(X_{i - 1} union i) - f(X_{i - 1}) and f(Y_{i - 1} \ {i}) - f(Y_{i - 1})
        a = f_diff(i, S)
        b = -f_diff(i, setdiff(Y, Set{Int64}(i)))

        #If a >= b, add the element to X; otherwise remove the element from Y
        if !random
            if a >= b
                push!(X, i)
            else
                setdiff!(Y, Set{Int64}(i))
            end
        else
            a = max(a, 0)
            b = max(b, 0)

            if a == 0 & b == 0
                push!(X, i)
            else
                r = rand()

                if r <= a / (a + b)
                    push!(X, i)
                else
                    setdiff!(Y, Set{Int64}(i))
                end
            end
        end
    end

    #Return X (or Y)
    return X
end

function sample_greedy(f_diff, ind, n, k; lazy = false)
    """
    #sample greedy algorithm (for k-extendible systems):
    #select a sample from the ground set with 1/(k + 1) probability of including each element
    #then run greedy on this sub-sample

    # Input
    # 	f_diff		a function to compute the marginal utility of an element e given a set S
    #   ind         a set of sets that are independent
    #   n           the size of the ground set
    #   k           ind should be a k-extendible system
    #
    # Output
    #   S
    """

    #First initialize the results
    S = Set{Int64}()
    sample = Set{Int64}()

    #Put every element into the sample with equal probability 1 / (k + 1)
    for i in 1 : n
        r = rand()

        if r <= 1 / (k + 1)
            push!(sample, i)
        end
    end

    #Run greedy on the sub-sample
    if !lazy

        while !isempty(sample)
            #Initiaze some variables to keep track of the best marginal gain and element so far
            max_index = 0
            max_so_far = -Inf
            cur = Set{Int64}()

            #Try to pick the best element to adjoin
            for j in sample
                if !independence(S, j, ind)
                    push!(cur, j)
                    continue
                else
                    cur_margin = f_diff(j, S)
                    if cur_margin > max_so_far
                        max_so_far = cur_margin
                        max_index = j
                    end
                end
            end

            #Adjoin the element only if the largest marginal utility is positive
            if max_so_far > 0
                push!(S, max_index)
                push!(cur, max_index)
                setdiff!(sample, cur)
            else
                return S
            end
        end
    else
        #Initialize a priority queue
        pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)

        #Initialize all the entries in the priority queue to their initial gain
        for j in sample
            enqueue!(pq, j, f_diff(j, S))
        end

        #Run the iterations until the priority queue is empty
        while !isempty(pq)
            cur_index, cur_margin = peek(pq)
            dequeue!(pq)

            #Return the set we have currently once the marginal utility becomes non-positive
            if cur_margin <= 0
                return S
            end

            #If adjoining the element breaks independence then just delete it from pq
            if !independence(S, cur_index, ind)
                continue
            else
                cur_margin = f_diff(cur_index, S)

                #If the updated marginal utility is non-positive then just delete it
                if cur_margin <= 0
                    continue
                else
                    if isempty(pq) || cur_margin >= peek(pq)[2]
                        push!(S, cur_index)
                    else
                        enqueue!(pq, cur_index, cur_margin)
                    end
                end
            end
        end
    end

    return S
end

function independence(S, e, ind)
    """
    #This will serve as an auxiliary function to sample greedy
    #For a given independent set S and e, check if adjoining e to S will break the independence

    # Input
    # 	S           the starting independent set
    #   e           an element in our consideration
    #   ind         a set of independent sets (a k-extendible system)
    #
    # Output
    #   a bool that indicates if the resulting set is independent
    """

    #Return true is e is already in the independent set S
    if e in S
        return true
    else
        #else just go over ind one by one to check if it is the resulting set
        S_2 = union(S, Set{Int64}(e))
        for A in ind
            if A == S_2
                return true
            end
        end
        return false
    end
end

function random_greedy(f_diff, n, k)
    """
    #A random greedy algorithm that approximates the optimal solution subject to a cardinality constraint

    # Input
    #   f_diff      a function that evaluates the marginal utility of
    # 	n           the number of elements in the ground set
    #   k           the cardinality constraint
    #
    # Output
    #   S           the set reulting from the greedy algorithm
    """

    #Return the ground set if the cardinality constraint is larger than or equal to the size of the ground set
    if k >= n
        return Set{Int64}(1 : n)
    end

    #Initialize the sets
    S = Set{Int64}()

    #Run k iterations of the algorithm
    for i in 1 : k
        #Use a priority queue in each iteration to select the best k elements
        pq = PriorityQueue{Int64, Float64}(Base.Order.Reverse)

        #Evaluate the marginal utility of each element not yet adjoined
        for j in 1 : n
            if in(S,j)
                continue
            end

            enqueue!(pq, j, f_diff(j, S))
        end

        #Pick out the best k elements and then select one of them uniformly at random
        cur = Set{Int64}()
        for h in 1 : k
            push!(cur, dequeue!(pq))
        end

        push!(S, rand(cur))
    end

    return S
end
