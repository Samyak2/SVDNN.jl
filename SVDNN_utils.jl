# Helper function to train one epoch - loss(data..., model) must be defined
function custom_train!(loss, ps, data, opt, model)
    @timeit to "Get params" ps = Params(ps)
    for d in data
        @timeit to "Calculate gradient" begin
            gs = gradient(ps) do
                training_loss = loss(d..., model)
                return training_loss
            end
        end
        @timeit to "Update params" update!(opt, ps, gs)
    end
end

# Helper function to train n epochs
# loss is the loss function
# cost_function calculates loss for all data and averages them
# accuracy is the function which returns model accuracy
# accuracies is an array to store train accuracy at each step
# test_accuracies is an array to store test accuracy at each step
function train_epochs!(loss, data, test_data, opt, model, epochs=10;
                       train_func=custom_train!,
                       cost_function=cost_function,
                       accuracy=accuracy,
                       accuracies=accuracies,
                       test_accuracies=test_accuracies)
    ps = params(model)
    for i in 1:epochs
        @timeit to "Train one step" train_func(loss, ps, data, opt, model)
        @timeit to "Calculate cost" cost = cost_function(data, loss, model)
        push!(costs, cost)
        @timeit to "Calculate accuracy" acc = accuracy(data, model)
        push!(accuracies, acc)
        @timeit to "Calculate test accuracy" test_acc = accuracy(test_data, model)
        push!(test_accuracies, test_acc)
        println("Cost at epoch $i is $cost. Accuracy $acc. Test accuracy $test_acc")
    end
end

# Helper function to train a SVD-DNN model for some epochs
# The model should be made using a chain of Dense layers (not SVDense)
# Given model will be pretrained for pretrain_percent of the epochs
# Then all Dense layers will be converted to SVDense (excepted those which have l value of "nothing")
# opt2 is the optimiser used for the newly created SVD-DNN model
# l_values is an array which denotes how many most significant singular values to keep,
# l_values should be less than the dimension at each layer, an l-value of "nothing" means that the layer
# will not be converted to a SVDense layer
#
# loss is the loss function
# cost_function calculates loss for all data and averages them
# accuracy is the function which returns model accuracy
# accuracies is an array to store train accuracy at each step
# test_accuracies is an array to store test accuracy at each step
function train_epochs_svd!(model, l_values, loss, data, test_data, opt, opt2, epochs=10, pretrain_percent=0.1;
                           train_func=custom_train!,
                           cost_function=cost_function,
                           accuracy=accuracy,
                           accuracies=accuracies,
                           test_accuracies=test_accuracies)
    @assert length(model) == length(l_values)

    ps = params(model)

    pretraining_epochs = round(Int, pretrain_percent * epochs)
    @timeit to "Pre-training" begin
        for i in 1:pretraining_epochs
            @timeit to "Train one epoch" train_func(loss, ps, data, opt, model)
            @timeit to "Calculate cost" cost = cost_function(data, loss, model)
            push!(costs, cost)
            @timeit to "Calculate accuracy" acc = accuracy(data, model)
            push!(accuracies, acc)
            @timeit to "Calculate test accuracy" test_acc = accuracy(test_data, model)
            push!(test_accuracies, test_acc)
            println("Cost at epoch $i is $cost. Accuracy $acc. Test accuracy $test_acc")
        end
    end
    println("Finished pre-training")

    @timeit to "Build SVD model" SVDmodel = Chain(
        map((layer, l) -> l == nothing ? layer : SVDense(layer, l),
            model,
            l_values)...
    )
    ps = params(SVDmodel)
    println("Built SVD model")

    acc = accuracy(data, SVDmodel)
    test_acc = accuracy(test_data, SVDmodel)
    println("SVD Model: Accuracy $acc. Test accuracy $test_acc")

    remaining_epochs = epochs - pretraining_epochs

    println("Begin tuning SVD model")
    @timeit to "Tuning SVD model" begin
        for i in 1:remaining_epochs
            @timeit to "[SVD] Train one epoch" train_func(loss, ps, data, opt2, SVDmodel)
            @timeit to "[SVD] Calculate cost" cost = cost_function(data, loss, SVDmodel)
            push!(costs, cost)
            @timeit to "[SVD] Calculate accuracy" acc = accuracy(data, SVDmodel)
            push!(accuracies, acc)
            @timeit to "[SVD] Calculate test accuracy" test_acc = accuracy(test_data, SVDmodel)
            push!(test_accuracies, test_acc)
            println("Cost at epoch $i is $cost. Accuracy $acc. Test accuracy $test_acc")
        end
    end

    @timeit to "Update weights of orig model" begin
        for (l_orig, l_new) in zip(model, SVDmodel)
            if l_new isa SVDense
                l_orig.W[:] = l_new.W1 * l_new.W2
                l_orig.b[:] = l_new.b
            else
                l_orig.W[:] = l_new.W
                l_orig.b[:] = l_new.b
            end
        end
    end

    println("Final accuracy $(accuracy(data, model))")
end
